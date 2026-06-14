"""
PerceptualGatekeeper v2: Multi-Head Cross-Attention Gate.

Key improvements over v1:
1. Kinematic gate uses multi-head cross-attention (not single-vector dot-product)
   - qpos → 8 query vectors (not 1) → CrossAttention with visual tokens
   - This gives the gate enough capacity for spatial selectivity
2. Works on ViT intermediate tokens (mid-layer pruning)
   - Pruning inside the backbone actually saves compute

Gate outputs:
  g_kin:    [B, N, 1]  Kinematic-guided gate (cross-attention based)
  g_vis:    [B, N, 1]  Visual saliency gate (self-attention importance)
  g_hybrid: [B, N, 1]  Fused gate = max(g_kin, g_vis)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import PolicyVariant, CONFIG


class PerceptualGatekeeper(nn.Module):
    def __init__(self, feature_dim, qpos_dim, num_gate_queries=8, num_heads=4, slow_semantic_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_gate_queries = num_gate_queries

        # ============================================================
        # Target-Centric Attention Gate (Option 2: Biomimetic Dynamic Filter)
        # ============================================================
        # PROMERGE_ONLY (Variant 4) learnable target queries
        self.default_target_queries = nn.Parameter(torch.randn(num_gate_queries, feature_dim))
        
        # Cross-attention for Variant 4
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.gate_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1)
        )

        # PROMERGE_FILM (Variant 5) dynamic intent filter generator
        self.intent_filter_proj = nn.Linear(slow_semantic_dim, feature_dim)

        # ============================================================
        # FiLM semantic modulation (for PROMERGE_FILM variant)
        # ============================================================
        self.film_generator = nn.Linear(slow_semantic_dim, feature_dim * 2)

        # Saved gate masks for visualization
        self.last_g_kin = None
        self.last_g_vis = None
        self.last_g_hybrid = None

    def reset_history(self):
        if hasattr(self, 'last_g_hybrid_smoothed'):
            delattr(self, 'last_g_hybrid_smoothed')

    def forward(self, visual_tokens, qpos, slow_semantic=None, pos_tokens=None):
        """
        Args:
            visual_tokens: [batch_size, N, D] intermediate ViT features (e.g. [batch_size, 1200, 384] per camera)
            qpos: [batch_size, qpos_dim]
            slow_semantic: [batch_size, slow_semantic_dim] (optional, for PROMERGE_FILM)
            pos_tokens: [batch_size, N, D] positional encodings (optional, for ViT we don't need external pos)

        Returns:
            If MONOLITHIC_ACT: visual_tokens unchanged (+ pos_tokens if provided)
            Otherwise: pruned_tokens [batch_size, keep_k, D] (+ pruned_pos if pos_tokens provided)
        """
        batch_size, N, C = visual_tokens.shape
        keep_k = int(N * CONFIG["keep_ratio"])

        # ==========================================
        # Variant 1: Monolithic ACT (no pruning)
        # ==========================================
        if CONFIG["variant"] == PolicyVariant.MONOLITHIC_ACT:
            if pos_tokens is not None:
                return visual_tokens, pos_tokens
            return visual_tokens

        # ==========================================
        # Variant 2: Random Pruning (baseline)
        # ==========================================
        elif CONFIG["variant"] == PolicyVariant.RANDOM_PRUNE:
            rand_indices = torch.stack(
                [torch.randperm(N, device=visual_tokens.device)[:keep_k] for _ in range(batch_size)], dim=0
            )
            pruned_visual = torch.gather(visual_tokens, 1, rand_indices.unsqueeze(-1).expand(-1, -1, C))
            if pos_tokens is not None:
                C_pos = pos_tokens.shape[-1]
                if pos_tokens.shape[0] == 1:
                    pos_tokens = pos_tokens.expand(batch_size, -1, -1)
                pruned_pos = torch.gather(pos_tokens, 1, rand_indices.unsqueeze(-1).expand(-1, -1, C_pos))
                return pruned_visual, pruned_pos
            return pruned_visual

        # ==========================================
        # Variant 3: ToMe Clustering (competitor baseline)
        # ==========================================
        elif CONFIG["variant"] == PolicyVariant.TOME_CLUSTERING:
            self_attn = torch.bmm(visual_tokens, visual_tokens.permute(0, 2, 1))  # [batch_size, N, N]
            importance = self_attn.sum(dim=-1)  # [batch_size, N]
            
            if CONFIG.get("merge_tokens", True):
                sorted_indices = torch.argsort(importance, dim=1, descending=True)
                topk_indices = sorted_indices[:, :keep_k]
                rest_indices = sorted_indices[:, keep_k:]
                
                A = torch.gather(visual_tokens, 1, topk_indices.unsqueeze(-1).expand(-1, -1, C))
                B = torch.gather(visual_tokens, 1, rest_indices.unsqueeze(-1).expand(-1, -1, C))
                
                A_norm = F.normalize(A, p=2, dim=-1)
                B_norm = F.normalize(B, p=2, dim=-1)
                sim = torch.bmm(B_norm, A_norm.transpose(1, 2))  # [batch_size, N-K, K]
                matched_indices = torch.argmax(sim, dim=-1)  # [batch_size, N-K]
                
                A_sum = A.clone()
                A_sum.scatter_add_(1, matched_indices.unsqueeze(-1).expand(-1, -1, C), B)
                w_A = torch.ones(batch_size, keep_k, 1, device=visual_tokens.device)
                w_B = torch.ones(batch_size, N - keep_k, 1, device=visual_tokens.device)
                w_A.scatter_add_(1, matched_indices.unsqueeze(-1), w_B)
                result_visual = A_sum / w_A
                topk_indices_final = topk_indices
            else:
                topk_indices = torch.topk(importance, keep_k, dim=1).indices
                result_visual = torch.gather(visual_tokens, 1, topk_indices.unsqueeze(-1).expand(-1, -1, C))
                topk_indices_final = topk_indices
                
            if pos_tokens is not None:
                C_pos = pos_tokens.shape[-1]
                if pos_tokens.shape[0] == 1:
                    pos_tokens = pos_tokens.expand(batch_size, -1, -1)
                pruned_pos = torch.gather(pos_tokens, 1, topk_indices_final.unsqueeze(-1).expand(-1, -1, C_pos))
                return result_visual, pruned_pos
            return result_visual

        # ==========================================
        # Variant 4 & 5: ProMerge (Cross-Attention Kinematic Gate)
        # ==========================================
        else:
            src_modulated = visual_tokens

            # 【Variant 5 only】: FiLM semantic modulation
            if CONFIG["variant"] == PolicyVariant.PROMERGE_FILM and slow_semantic is not None:
                film_params = self.film_generator(slow_semantic).unsqueeze(1)  # [batch_size, 1, 2*C]
                gamma, beta = torch.chunk(film_params, 2, dim=-1)
                src_modulated = visual_tokens * (1 + gamma) + beta

            # ---- g_kin: Target-Centric Gate ----
            if CONFIG["variant"] == PolicyVariant.PROMERGE_FILM:
                # Option 2 (biomimetic): Generate target filter w from slow_semantic
                if slow_semantic is not None:
                    w = self.intent_filter_proj(slow_semantic)  # [batch_size, D]
                    # Compute dot product: V [batch_size, N, D] * w [batch_size, D, 1] -> [batch_size, N]
                    scores = torch.bmm(src_modulated, w.unsqueeze(-1)).squeeze(-1)
                    g_kin = torch.sigmoid(scores / math.sqrt(C)).unsqueeze(-1)  # [batch_size, N, 1]
                else:
                    g_kin = torch.ones(batch_size, N, 1, device=visual_tokens.device)
            else:
                # Variant 4 (PROMERGE_ONLY): Static Learnable Target Queries (no qpos)
                queries = self.default_target_queries.unsqueeze(0).expand(batch_size, -1, -1)
                
                attn_out, attn_weights = self.cross_attn(
                    query=queries,
                    key=src_modulated,
                    value=src_modulated
                )
                token_importance = attn_weights.max(dim=1).values  # [batch_size, N]
                
                ti_min = token_importance.min(dim=1, keepdim=True).values
                ti_max = token_importance.max(dim=1, keepdim=True).values
                token_importance_normalized = (token_importance - ti_min) / (ti_max - ti_min + 1e-8)
                
                gate_input = src_modulated * token_importance_normalized.unsqueeze(-1)
                g_kin = torch.sigmoid(self.gate_head(gate_input))  # [batch_size, N, 1]

            # ---- g_vis: Visual Saliency Gate ----
            self_attn = torch.bmm(src_modulated, src_modulated.permute(0, 2, 1))  # [batch_size, N, N]
            importance = self_attn.sum(dim=-1).unsqueeze(-1)  # [batch_size, N, 1]
            imp_min = importance.min(dim=1, keepdim=True).values
            imp_max = importance.max(dim=1, keepdim=True).values
            g_vis = (importance - imp_min) / (imp_max - imp_min + 1e-8)  # [batch_size, N, 1]

            # ---- Hybrid Gate Fusion ----
            g_hybrid = torch.max(g_kin, g_vis)  # [batch_size, N, 1]

            # Apply temporal smoothing only during inference (eval mode)
            if not self.training:
                if hasattr(self, 'last_g_hybrid_smoothed') and self.last_g_hybrid_smoothed.shape == g_hybrid.shape:
                    g_hybrid = 0.7 * g_hybrid + 0.3 * self.last_g_hybrid_smoothed.to(g_hybrid.device)
                self.last_g_hybrid_smoothed = g_hybrid.detach()

            # Save for visualization
            self.last_g_kin = g_kin.detach().cpu()
            self.last_g_vis = g_vis.detach().cpu()
            self.last_g_hybrid = g_hybrid.detach().cpu()

            # ---- Soft Gating: Apply clamp to prevent complete background blackout ----
            gate_mask = torch.clamp(g_hybrid, min=0.05)  # [batch_size, N, 1]
            src_modulated = src_modulated * gate_mask

            # ---- Spatial Dilution (Morphological Dilatation) on the 15x20 Grid ----
            img_size = CONFIG.get("image_size", (480, 640))
            if CONFIG.get("backbone") == "vit_small":
                H_feat = img_size[0] // 16
                W_feat = img_size[1] // 16
            else:
                H_feat = img_size[0] // 32
                W_feat = img_size[1] // 32
            
            num_cameras = N // (H_feat * W_feat)
            
            # Reshape scores to 2D: [batch_size * num_cameras, 1, H_feat, W_feat]
            scores_2d = g_hybrid.view(batch_size * num_cameras, 1, H_feat, W_feat)
            
            # Perform 1-step spatial dilation via MaxPool2d
            scores_dilated_2d = F.max_pool2d(scores_2d, kernel_size=3, stride=1, padding=1)
            
            # Reshape back to [batch_size, N]
            scores_dilated = scores_dilated_2d.view(batch_size, N)

            # ---- Top-K Token Selection/Merging on Dilated Scores ----
            if CONFIG.get("merge_tokens", True):
                sorted_indices = torch.argsort(scores_dilated, dim=1, descending=True)
                topk_indices = sorted_indices[:, :keep_k]
                rest_indices = sorted_indices[:, keep_k:]
                
                A = torch.gather(src_modulated, 1, topk_indices.unsqueeze(-1).expand(-1, -1, C))
                B = torch.gather(src_modulated, 1, rest_indices.unsqueeze(-1).expand(-1, -1, C))
                
                A_norm = F.normalize(A, p=2, dim=-1)
                B_norm = F.normalize(B, p=2, dim=-1)
                sim = torch.bmm(B_norm, A_norm.transpose(1, 2))  # [batch_size, N-K, K]
                matched_indices = torch.argmax(sim, dim=-1)  # [batch_size, N-K]
                
                A_sum = A.clone()
                A_sum.scatter_add_(1, matched_indices.unsqueeze(-1).expand(-1, -1, C), B)
                w_A = torch.ones(batch_size, keep_k, 1, device=visual_tokens.device)
                w_B = torch.ones(batch_size, N - keep_k, 1, device=visual_tokens.device)
                w_A.scatter_add_(1, matched_indices.unsqueeze(-1), w_B)
                result_visual = A_sum / w_A
                topk_indices_final = topk_indices
            else:
                topk_indices = torch.topk(scores_dilated, keep_k, dim=1).indices
                result_visual = torch.gather(src_modulated, 1, topk_indices.unsqueeze(-1).expand(-1, -1, C))
                topk_indices_final = topk_indices

            if pos_tokens is not None:
                C_pos = pos_tokens.shape[-1]
                if pos_tokens.shape[0] == 1:
                    pos_tokens = pos_tokens.expand(batch_size, -1, -1)
                pruned_pos = torch.gather(pos_tokens, 1, topk_indices_final.unsqueeze(-1).expand(-1, -1, C_pos))
                return result_visual, pruned_pos
            return result_visual
