"""
ViT-Small Backbone with Mid-Layer Token Pruning Support.

This replaces ResNet18 as the visual backbone for ProMerge.
Key design: token pruning happens INSIDE the ViT (between layers),
so reducing tokens actually saves compute in subsequent layers.

Architecture:
  Image [B, 3, 480, 640]
    → PatchEmbed → [B, 1200, 384]  (30×40 patches of 16×16)
    → ViT Blocks 0..pruning_layer-1 (full token computation)
    → [EXTERNAL: Gatekeeper prunes tokens here]
    → ViT Blocks pruning_layer..11 (pruned token computation)
    → Output: [B, N_pruned, 384]
"""
import torch
import torch.nn as nn
import timm
from config import CONFIG


class ViTBackbone(nn.Module):
    def __init__(self, img_size=(480, 640), pruning_layer=6, pretrained=True):
        super().__init__()
        self.pruning_layer = pruning_layer
        self.img_size = img_size

        # Create ViT-Small from timm with custom image size
        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,   # remove classification head
            global_pool='',  # no pooling, return all tokens
        )

        self.embed_dim = self.vit.embed_dim  # 384
        self.num_patches = self.vit.patch_embed.num_patches  # 30*40 = 1200
        self.num_blocks = len(self.vit.blocks)  # 12

        # Spatial dimensions of the patch grid
        self.grid_h = img_size[0] // 16  # 30
        self.grid_w = img_size[1] // 16  # 40

        # Property for compatibility
        self.num_channels = self.embed_dim  # 384

    def forward_first_half(self, x):
        """
        Run patch embedding + first half of ViT blocks.
        
        Args:
            x: [B, 3, H, W] image tensor
            
        Returns:
            tokens: [B, N, D] intermediate visual tokens (no CLS token)
                    N = num_patches = 1200, D = 384
        """
        # Patch embedding
        tokens = self.vit.patch_embed(x)  # [B, 1200, 384]

        # Add positional embedding (skip CLS position)
        tokens = tokens + self.vit.pos_embed[:, 1:, :]

        # We intentionally do NOT use CLS token during feature extraction
        # to keep tokens purely spatial for the gatekeeper
        tokens = self.vit.pos_drop(tokens)

        # Run first half of ViT blocks
        for i in range(self.pruning_layer):
            tokens = self.vit.blocks[i](tokens)

        return tokens  # [B, 1200, 384]

    def forward_second_half(self, tokens):
        """
        Run second half of ViT blocks on (potentially pruned) tokens.
        
        Args:
            tokens: [B, N_pruned, D] pruned visual tokens
            
        Returns:
            tokens: [B, N_pruned, D] final visual features
        """
        for i in range(self.pruning_layer, self.num_blocks):
            tokens = self.vit.blocks[i](tokens)

        # Apply final norm
        tokens = self.vit.norm(tokens)

        return tokens  # [B, N_pruned, 384]

    def forward(self, x):
        """Full forward pass without pruning (for MONOLITHIC_ACT baseline)."""
        tokens = self.forward_first_half(x)
        tokens = self.forward_second_half(tokens)
        return tokens


class CLIPViTBackbone(nn.Module):
    """CLIP ViT-B/32 visual backbone with the same mid-layer-pruning interface
    as ViTBackbone (forward_first_half / forward_second_half / forward).

    Why: the instruction is encoded with CLIP's TEXT encoder, but the original
    visual backbone was an ImageNet-pretrained timm ViT — a DIFFERENT embedding
    space. Intent grounding (g_kin) failed because CLIP-text and ImageNet-ViT
    features are not aligned. Using CLIP's VISION encoder puts text and vision
    in the SAME CLIP space, so dot-product / cross-attention grounding becomes
    meaningful.

    Notes / differences vs the timm ViT:
      - embed_dim = 768 (vs 384) -> downstream hidden_dim must match (see config).
      - patch32 @ 224 -> 49 patch tokens (vs 1200). Far fewer tokens already.
      - CLS token is dropped to keep tokens purely spatial for the gatekeeper.
    """
    def __init__(self, img_size=(224, 224), pruning_layer=6, pretrained=True,
                 model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        from transformers import CLIPVisionModel
        self.pruning_layer = pruning_layer
        self.img_size = img_size
        self.clip = CLIPVisionModel.from_pretrained(model_name)
        # FREEZE the CLIP vision encoder: fine-tuning it would drift its features
        # away from the CLIP TEXT space, destroying the very text<->vision
        # alignment we adopted CLIP for. Only the downstream projection /
        # gatekeeper / ACT are trained.
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()
        self.embed_dim = self.clip.config.hidden_size          # 768
        self.num_blocks = self.clip.config.num_hidden_layers   # 12
        ps = self.clip.config.patch_size                       # 32
        self.grid_h = img_size[0] // ps                        # 7
        self.grid_w = img_size[1] // ps                        # 7
        self.num_patches = self.grid_h * self.grid_w           # 49
        self.num_channels = self.embed_dim

    def _embed(self, x):
        vm = self.clip
        # CLIP expects 224x224; resize if needed
        if x.shape[-2:] != tuple(self.img_size):
            x = nn.functional.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        h = vm.embeddings(x)            # [B, 1+num_patches, D]  (CLS + patches)
        h = vm.pre_layrnorm(h)
        return h

    def forward_first_half(self, x):
        h = self._embed(x)             # [B, 1+N, D]
        for i in range(self.pruning_layer):
            h = self.clip.encoder.layers[i](h, attention_mask=None)
        # drop CLS token -> purely spatial tokens for the gatekeeper
        return h[:, 1:, :]             # [B, N, D]

    def forward_second_half(self, tokens):
        # tokens are spatial-only (CLS already removed / pruned); CLIP layers
        # operate on the sequence as-is.
        h = tokens
        for i in range(self.pruning_layer, self.num_blocks):
            h = self.clip.encoder.layers[i](h, attention_mask=None)
        h = self.clip.post_layernorm(h)
        return h

    def forward(self, x):
        tokens = self.forward_first_half(x)
        tokens = self.forward_second_half(tokens)
        return tokens
