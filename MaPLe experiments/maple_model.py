"""
MaPLe (Multi-modal Prompt Learning) Model Implementation
=========================================================
Based on: "MaPLe: Multi-modal Prompt Learning" (CVPR 2023)
Paper: https://arxiv.org/abs/2210.03117

MaPLe extends CoOp by:
1. Injecting learnable prompts into BOTH vision and text encoders
2. Using deep prompts that propagate through multiple transformer layers
3. Coupling vision-language prompts via a projection network

Key Differences from CoOp:
- CoOp: Only text encoder prompts at layer 0
- MaPLe: Both vision and text prompts at layers 0 to J-1 (typically J=9)
"""

import torch
import torch.nn as nn
import clip
from collections import OrderedDict


class VLCoupler(nn.Module):
    """
    Vision-Language Coupler: Projects text prompts to vision prompt space.
    This maintains multi-modal alignment by deriving vision prompts from text prompts.
    """
    def __init__(self, text_dim, vision_dim):
        super().__init__()
        self.proj = nn.Linear(text_dim, vision_dim)

    def forward(self, text_prompts):
        """
        Args:
            text_prompts: (n_ctx, text_dim) - learnable text prompt embeddings
        Returns:
            vision_prompts: (n_ctx, vision_dim) - projected vision prompt embeddings
        """
        return self.proj(text_prompts)


class DeepTextEncoder(nn.Module):
    """
    Modified CLIP text encoder that accepts deep prompts at multiple layers.
    We monkey-patch the transformer to inject prompts before each attention block.
    """
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.n_layers = len(self.transformer.resblocks)

    def forward(self, prompts, tokenized_prompts, deep_prompts=None, prompt_depth=0):
        """
        Args:
            prompts: (n_cls, seq_len, ctx_dim) - embeddings with learnable context at layer 0
            tokenized_prompts: (n_cls, seq_len) - token IDs for EOS position
            deep_prompts: list of (n_ctx, ctx_dim) tensors for layers 1 to prompt_depth-1
            prompt_depth: number of layers to inject prompts (default: use all)
        """
        x = prompts.type(self.dtype) + self.positional_embedding[:prompts.size(1), :].type(self.dtype)
        x = x.permute(1, 0, 2)  # (seq_len, batch, dim)

        # Run transformer layers with optional deep prompt injection.
        for i, block in enumerate(self.transformer.resblocks):
            if i > 0 and deep_prompts is not None and i <= len(deep_prompts):
                # Get deep prompts for this layer.
                layer_prompts = deep_prompts[i - 1]  # (n_ctx, ctx_dim)
                n_ctx = layer_prompts.shape[0]
                n_cls = x.shape[1]

                # Expand for all classes: (n_ctx, n_cls, ctx_dim)
                layer_prompts = layer_prompts.unsqueeze(1).expand(-1, n_cls, -1)

                # Replace the first n_ctx tokens (after SOS) with deep prompts
                # x shape: (seq_len, batch, dim) where seq_len = 1(SOS) + n_ctx + suffix
                prefix = x[:1, :, :]  # SOS token
                suffix = x[1 + n_ctx:, :, :]  # Class tokens + EOS + padding
                x = torch.cat([prefix, layer_prompts.type(self.dtype), suffix], dim=0)

            x = block(x)

        x = x.permute(1, 0, 2)  # (batch, seq_len, dim)
        x = self.ln_final(x).type(self.dtype)

        # Extract features at the EOS position.
        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)]
        x = x @ self.text_projection

        return x


class DeepVisionEncoder(nn.Module):
    """
    Modified CLIP vision encoder that accepts deep prompts at multiple layers.
    Injects vision prompts at the beginning of each transformer layer.
    """
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self.dtype = clip_model.dtype
        self.n_layers = len(self.visual.transformer.resblocks)

    def forward(self, images, deep_prompts=None, prompt_depth=0):
        """
        Args:
            images: (B, 3, H, W) - input images
            deep_prompts: list of (n_ctx, vision_dim) tensors for each layer
            prompt_depth: number of layers to inject prompts
        """
        # Initial patch embedding.
        x = self.visual.conv1(images.type(self.dtype))  # (B, D, H/P, W/P)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, D, N_patches)
        x = x.permute(0, 2, 1)  # (B, N_patches, D)

        # Prepend the class token.
        cls_token = self.visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)  # (B, 1 + N_patches, D)

        # Add positional embeddings.
        x = x + self.visual.positional_embedding.to(x.dtype)

        # Apply pre-layer normalization.
        x = self.visual.ln_pre(x)

        # Inject prompts at layer 0 when provided.
        if deep_prompts is not None and len(deep_prompts) > 0:
            n_ctx = deep_prompts[0].shape[0]
            batch_size = x.shape[0]

            # Expand prompts for batch: (n_ctx, vision_dim) -> (B, n_ctx, vision_dim)
            layer_prompts = deep_prompts[0].unsqueeze(0).expand(batch_size, -1, -1)

            # Insert prompts after CLS token: [CLS, prompts, patches]
            cls_token = x[:, :1, :]  # (B, 1, D)
            patch_tokens = x[:, 1:, :]  # (B, N_patches, D)
            x = torch.cat([cls_token, layer_prompts.type(self.dtype), patch_tokens], dim=1)

        # Convert NLD -> LND for the transformer.
        x = x.permute(1, 0, 2)

        # Run transformer layers.
        for i, block in enumerate(self.visual.transformer.resblocks):
            if i > 0 and deep_prompts is not None and i < len(deep_prompts):
                # Inject deep prompts at this layer.
                layer_prompts = deep_prompts[i]  # (n_ctx, vision_dim)
                n_ctx = layer_prompts.shape[0]
                batch_size = x.shape[1]

                # Expand prompts for the batch.
                layer_prompts = layer_prompts.unsqueeze(1).expand(-1, batch_size, -1)

                # Replace the prompt tokens (after CLS)
                # x shape: (seq_len, batch, dim)
                cls_token = x[:1, :, :]  # CLS token
                remaining = x[1 + n_ctx:, :, :]  # Patch tokens (skip old prompts)
                x = torch.cat([cls_token, layer_prompts.type(self.dtype), remaining], dim=0)

            x = block(x)

        # Convert LND -> NLD.
        x = x.permute(1, 0, 2)

        # Extract the CLS token representation.
        x = self.visual.ln_post(x[:, 0, :])

        # Project to the shared space.
        if self.visual.proj is not None:
            x = x @ self.visual.proj

        return x


class MaPLePromptLearner(nn.Module):
    """
    MaPLe Prompt Learner: Manages learnable prompts for both vision and text encoders
    across multiple transformer layers.

    Key components:
    1. text_ctx: Learnable text prompts at layer 0
    2. deep_text_prompts: Learnable text prompts for layers 1 to J-1
    3. vl_coupler: Projects text prompts to vision prompts (maintains alignment)
    """
    def __init__(self, classnames, clip_model, n_ctx=2, prompt_depth=9, ctx_init=""):
        super().__init__()
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.prompt_depth = prompt_depth  # Number of layers to use deep prompts

        # Read embedding dimensions.
        ctx_dim = clip_model.ln_final.weight.shape[0]  # Text embedding dim (512 for ViT-B/16)
        vis_dim = clip_model.visual.conv1.weight.shape[0]  # Vision embedding dim (768 for ViT-B/16)

        self.ctx_dim = ctx_dim
        self.vis_dim = vis_dim

        print(f"MaPLe Config: n_ctx={n_ctx}, prompt_depth={prompt_depth}")
        print(f"  Text dim: {ctx_dim}, Vision dim: {vis_dim}")

        device = clip_model.ln_final.weight.device

        # Initialize layer-0 text prompts.
        if ctx_init:
            ctx_init_text = ctx_init.replace("_", " ")
            n_ctx_words = len(ctx_init_text.split())
            if n_ctx_words != n_ctx:
                print(f"  Warning: ctx_init has {n_ctx_words} words but n_ctx={n_ctx}")
                n_ctx = n_ctx_words
                self.n_ctx = n_ctx
            print(f"  Initializing with: '{ctx_init}'")
            prompt_tokens = clip.tokenize(ctx_init_text).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt_tokens)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
        else:
            print(f"  Random initialization")
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)

        # Learnable text context at layer 0.
        self.ctx = nn.Parameter(ctx_vectors)

        # Initialize deep text prompts for layers 1 to prompt_depth-1.
        if prompt_depth > 1:
            deep_prompts = []
            for _ in range(prompt_depth - 1):
                vectors = torch.empty(n_ctx, ctx_dim)
                nn.init.normal_(vectors, std=0.02)
                deep_prompts.append(nn.Parameter(vectors))
            self.deep_text_prompts = nn.ParameterList(deep_prompts)
        else:
            self.deep_text_prompts = None

        # Use the vision-language coupler to project text prompts into vision space.
        # Create one coupler per layer.
        self.vl_couplers = nn.ModuleList([
            VLCoupler(ctx_dim, vis_dim) for _ in range(prompt_depth)
        ])

        # Process class names for prompt suffixes.
        classnames = [name.replace("_", " ") for name in classnames]
        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [f"{prompt_prefix} {name}." for name in classnames]

        tokenized_prompts = clip.tokenize(prompts)  # (n_cls, 77)
        tokenized_prompts = tokenized_prompts.to(device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)

        # Store frozen prefix (SOS) and suffix (class + EOS)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.tokenized_prompts = tokenized_prompts

    def forward(self):
        """
        Returns:
            text_prompts: (n_cls, seq_len, ctx_dim) - complete text prompts at layer 0
            deep_text_prompts: list of (n_ctx, ctx_dim) for layers 1+
            vision_prompts: list of (n_ctx, vis_dim) for all layers
        """
        # Build layer-0 text prompts.
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # (n_cls, n_ctx, ctx_dim)
        text_prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

        # Collect text prompts for vision coupling.
        all_text_prompts = [self.ctx]  # Layer 0
        if self.deep_text_prompts is not None:
            all_text_prompts.extend(list(self.deep_text_prompts))

        # Generate vision prompts through coupling.
        vision_prompts = []
        for i, (text_prompt, coupler) in enumerate(zip(all_text_prompts, self.vl_couplers)):
            vision_prompt = coupler(text_prompt)  # (n_ctx, vis_dim)
            vision_prompts.append(vision_prompt)

        # Collect deep text prompts for layers 1+.
        deep_text = list(self.deep_text_prompts) if self.deep_text_prompts is not None else None

        return text_prompts, deep_text, vision_prompts


class CustomCLIPMaPLe(nn.Module):
    """
    MaPLe wrapper that combines:
    - Frozen CLIP backbone
    - MaPLe Prompt Learner (multi-modal deep prompts)
    - Modified text and vision encoders for deep prompt injection
    """
    def __init__(self, classnames, clip_model, n_ctx=2, prompt_depth=9, ctx_init=""):
        super().__init__()

        # Create the prompt learner.
        self.prompt_learner = MaPLePromptLearner(
            classnames, clip_model, n_ctx, prompt_depth, ctx_init
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        # Create encoders with deep prompt injection support.
        self.text_encoder = DeepTextEncoder(clip_model)
        self.image_encoder = DeepVisionEncoder(clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.prompt_depth = prompt_depth

    def forward(self, images):
        """
        Forward pass with deep multi-modal prompts.
        """
        # Get prompts from the learner.
        text_prompts, deep_text_prompts, vision_prompts = self.prompt_learner()

        # Encode images with vision prompts.
        image_features = self.image_encoder(
            images,
            deep_prompts=vision_prompts,
            prompt_depth=self.prompt_depth
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Encode text with deep prompts.
        text_features = self.text_encoder(
            text_prompts,
            self.tokenized_prompts,
            deep_prompts=deep_text_prompts,
            prompt_depth=self.prompt_depth
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute logits.
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def get_text_features(self):
        """Get text features for all classes (useful for analysis)."""
        text_prompts, deep_text_prompts, _ = self.prompt_learner()
        text_features = self.text_encoder(
            text_prompts,
            self.tokenized_prompts,
            deep_prompts=deep_text_prompts,
            prompt_depth=self.prompt_depth
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features


def build_maple_model(classnames, clip_model, n_ctx=2, prompt_depth=9, ctx_init=""):
    """
    Factory function to create a MaPLe model.

    Args:
        classnames: list of class names
        clip_model: loaded CLIP model
        n_ctx: number of context tokens (default: 2, as per MaPLe paper)
        prompt_depth: number of layers for deep prompts (default: 9)
        ctx_init: optional initialization string

    Returns:
        CustomCLIPMaPLe model
    """
    model = CustomCLIPMaPLe(classnames, clip_model, n_ctx, prompt_depth, ctx_init)
    return model


# Base-to-new generalization utilities.

def split_classes_base_new(classnames, seed=1):
    """
    Split classes into base (seen) and new (unseen) for generalization evaluation.

    Args:
        classnames: list of class names
        seed: random seed for reproducible splits

    Returns:
        base_classes: first half of classes (for training)
        new_classes: second half of classes (for evaluation)
        base_indices: indices of base classes
        new_indices: indices of new classes
    """
    import random
    random.seed(seed)

    n_classes = len(classnames)
    n_base = n_classes // 2

    # Shuffle class indices.
    indices = list(range(n_classes))
    random.shuffle(indices)

    base_indices = sorted(indices[:n_base])
    new_indices = sorted(indices[n_base:])

    base_classes = [classnames[i] for i in base_indices]
    new_classes = [classnames[i] for i in new_indices]

    return base_classes, new_classes, base_indices, new_indices


def compute_harmonic_mean(base_acc, new_acc):
    """
    Compute harmonic mean of base and new accuracies.
    This is the "gold standard" metric for PEFT generalization.

    H = 2 * base * new / (base + new)
    """
    if base_acc + new_acc == 0:
        return 0.0
    return 2 * base_acc * new_acc / (base_acc + new_acc)
