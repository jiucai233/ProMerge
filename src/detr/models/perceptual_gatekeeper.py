import os
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


def rms_norm(x, eps=1e-6):
    """RMSNorm without learnable gain (as used in ThinkProprio's selector)."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class PerceptualGatekeeper(nn.Module):
    def __init__(self, feature_dim, qpos_dim, num_gate_queries=8, num_heads=4, slow_semantic_dim=384):
        super().__init__()
        self.feature_dim = feature_dim
        self.qpos_dim = qpos_dim
        self.num_gate_queries = num_gate_queries

        # ============================================================
        # ThinkProprio baseline (faithful reimplementation of Sec 3.3)
        # Guidance H_q = [H_l (instruction); H_p (text-tokenized proprio)].
        # ============================================================
        self.tp_num_bins = CONFIG.get("tp_num_bins", 256)
        # Proprio "text" tokenization: discretize each qpos dim -> learnable embedding
        # (local analogue of reusing the VLM vocab embedding table; no VLM available here).
        self.tp_proprio_embed = nn.Embedding(self.tp_num_bins, feature_dim)
        # Instruction guidance token: project slow_semantic into the token space.
        self.tp_lang_proj = nn.Linear(slow_semantic_dim, feature_dim)
        # Annealed Gumbel temperature schedule (cosine), driven by a forward-step counter.
        self.register_buffer("tp_step", torch.zeros(1, dtype=torch.long), persistent=False)
        self.tp_alpha_start = CONFIG.get("tp_alpha_start", 1.0)
        self.tp_alpha_end = CONFIG.get("tp_alpha_end", 0.01)
        self.tp_anneal_steps = CONFIG.get("tp_anneal_steps", 5000)

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
        # Small-init the intent filter (PROMERGE_INIT=small) so g_kin starts near
        # uniform (~0.5 after sigmoid) instead of a random direction that injects
        # noisy gating from step 0. Random init is the suspected cause of the high
        # seed-to-seed variance (53-75% across seeds). Test if small init reduces it.
        if os.environ.get("PROMERGE_INIT", "default") == "small":
            nn.init.normal_(self.intent_filter_proj.weight, std=1e-3)
            nn.init.zeros_(self.intent_filter_proj.bias)

        # Cross-attention intent grounding (PROMERGE_FILM, gkin_mode="xattn").
        # Replaces the bilinear dot-product (which collapses the instruction-vs-
        # token relation into a single scalar and cannot ground language to the
        # right object) with multi-head cross-attention: the instruction is a
        # query attending over visual tokens, and the attention weights become
        # the per-token intent gate g_kin. This preserves structured, non-linear
        # text<->vision alignment instead of a lossy linear projection.
        self.intent_q_proj = nn.Linear(slow_semantic_dim, feature_dim)
        self.intent_xattn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )

        # ============================================================
        # FiLM semantic modulation (for PROMERGE_FILM variant)
        # ============================================================
        self.film_generator = nn.Linear(slow_semantic_dim, feature_dim * 2)
        # Zero-init so FiLM starts as the identity transform:
        # src_modulated = visual_tokens * (1 + 0) + 0 = visual_tokens.
        # Without this the randomly-initialized gamma/beta scramble the visual
        # tokens from step 0, which prevented PROMERGE_FILM from converging
        # (gripper output collapsed to a constant -> 0% success).
        nn.init.zeros_(self.film_generator.weight)
        nn.init.zeros_(self.film_generator.bias)

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
        # Variant 6: ThinkProprio baseline (vote-based hard selection, Sec 3.3)
        # ==========================================
        elif CONFIG["variant"] == PolicyVariant.THINKPROPRIO:
            return self._thinkproprio_select(
                visual_tokens, qpos, slow_semantic, pos_tokens, keep_k
            )

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
                if slow_semantic is not None:
                    gkin_mode = os.environ.get("PROMERGE_GKIN", "dot")
                    if gkin_mode == "xattn":
                        # Cross-attention grounding: instruction = query, visual
                        # tokens = key/value. The attention weights (query over
                        # all N tokens) are the per-token intent gate.
                        q = self.intent_q_proj(slow_semantic).unsqueeze(1)   # [B, 1, D]
                        _, attn_w = self.intent_xattn(
                            q, src_modulated, src_modulated,
                            need_weights=True, average_attn_weights=True,
                        )  # attn_w: [B, 1, N]
                        a = attn_w.squeeze(1)                                # [B, N]
                        # normalize to [0,1] per sample so it acts as a gate
                        a_min = a.min(dim=1, keepdim=True).values
                        a_max = a.max(dim=1, keepdim=True).values
                        g_kin = ((a - a_min) / (a_max - a_min + 1e-8)).unsqueeze(-1)  # [B, N, 1]
                    else:
                        # Legacy bilinear dot-product (lossy; kept for ablation).
                        w = self.intent_filter_proj(slow_semantic)  # [batch_size, D]
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
            # NOTE: self_attn.sum() measures how SIMILAR a token is to all others
            # = "background homogeneity", NOT saliency. Big homogeneous regions
            # (the dark cabinet, the table) score HIGH, while a unique target
            # object (the small black bowl) scores LOW — the opposite of saliency.
            # PROMERGE_GVIS=uniqueness flips this: a token is salient when it is
            # DISSIMILAR from the rest (cosine-based uniqueness).
            gvis_mode = os.environ.get("PROMERGE_GVIS", "sum")
            if gvis_mode == "uniqueness":
                vn = F.normalize(src_modulated, p=2, dim=-1)                 # cosine space
                sim = torch.bmm(vn, vn.permute(0, 2, 1))                     # [B, N, N] in [-1,1]
                mean_sim = sim.mean(dim=-1, keepdim=True)                    # avg similarity to others
                importance = (-mean_sim)                                    # unique = low avg sim
            else:
                self_attn = torch.bmm(src_modulated, src_modulated.permute(0, 2, 1))
                importance = self_attn.sum(dim=-1).unsqueeze(-1)            # legacy (background homogeneity)
            imp_min = importance.min(dim=1, keepdim=True).values
            imp_max = importance.max(dim=1, keepdim=True).values
            g_vis = (importance - imp_min) / (imp_max - imp_min + 1e-8)  # [batch_size, N, 1]

            # ---- Hybrid Gate Fusion ----
            # Diagnostic switches (eval-only, env-controlled) to attribute the
            # contribution of the intent gate vs. the saliency gate:
            #   PROMERGE_GATE=vis  -> use only g_vis (pure saliency, ToMe-like)
            #   PROMERGE_GATE=kin  -> use only g_kin (pure intent)
            #   (default)          -> max(g_kin, g_vis)
            _gate_mode = os.environ.get("PROMERGE_GATE", "hybrid")
            if _gate_mode == "vis":
                g_hybrid = g_vis
            elif _gate_mode == "kin":
                g_hybrid = g_kin
            elif _gate_mode == "w82":          # intent-dominant weighted sum
                g_hybrid = 0.8 * g_kin + 0.2 * g_vis
            elif _gate_mode == "w91":
                g_hybrid = 0.9 * g_kin + 0.1 * g_vis
            elif _gate_mode == "mult":         # multiplicative: saliency modulates intent
                g_hybrid = g_kin * (0.5 + 0.5 * g_vis)
            elif _gate_mode == "kin_visfloor": # intent leads, saliency only as a weak floor
                g_hybrid = torch.maximum(g_kin, 0.3 * g_vis)
            else:
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

            # ---- Spatial Dilution (Morphological Dilatation) ----
            # Infer the per-camera grid from the token count instead of hardcoding
            # img_size//patch (which breaks for CLIP-B/32: 49 tokens = 7x7).
            img_size = CONFIG.get("image_size", (480, 640))
            patch = 16 if CONFIG.get("backbone") == "vit_small" else (
                32 if CONFIG.get("backbone") == "clip_vit" else 32)
            H_feat = img_size[0] // patch
            W_feat = img_size[1] // patch
            num_cameras = max(1, N // (H_feat * W_feat))
            if num_cameras * H_feat * W_feat != N:
                # Grid doesn't divide N (e.g. CLIP square grid): derive a square
                # grid per camera from N. If still not square, skip dilation.
                ncam = len(CONFIG.get("camera_names", ["front", "wrist"])) if CONFIG.get("camera_names") else 2
                per_cam = N // ncam
                side = int(round(per_cam ** 0.5))
                if side * side == per_cam:
                    num_cameras, H_feat, W_feat = ncam, side, side
                else:
                    num_cameras = None  # skip dilation

            if num_cameras is not None:
                scores_2d = g_hybrid.view(batch_size * num_cameras, 1, H_feat, W_feat)
                scores_dilated_2d = F.max_pool2d(scores_2d, kernel_size=3, stride=1, padding=1)
                scores_dilated = scores_dilated_2d.view(batch_size, N)
            else:
                scores_dilated = g_hybrid.view(batch_size, N)  # no dilation

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

    def _build_proprio_tokens(self, qpos):
        """Text-tokenize proprioception (ThinkProprio Sec 3.2 analogue).

        qpos is already z-normalized upstream, so the paper's [-3, 3] clip range maps
        naturally onto the standardized state. Each dim -> a bin index -> a learnable
        embedding (local stand-in for the VLM token embedding table).
        """
        q = torch.clamp(qpos, -3.0, 3.0)
        bins = torch.floor((q + 3.0) / 6.0 * self.tp_num_bins)
        bins = torch.clamp(bins, 0, self.tp_num_bins - 1).long()  # [B, p]
        return self.tp_proprio_embed(bins)  # [B, p, D]

    def _current_gumbel_alpha(self):
        if self.tp_anneal_steps <= 0:
            return self.tp_alpha_end
        t = float(self.tp_step.item()) / float(self.tp_anneal_steps)
        t = min(max(t, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 -> 0
        return self.tp_alpha_end + (self.tp_alpha_start - self.tp_alpha_end) * cos

    def _thinkproprio_select(self, visual_tokens, qpos, slow_semantic, pos_tokens, keep_k):
        """Physically grounded vote-based token selection with Gumbel + STE (Sec 3.3).

        Deviation from the paper, for batched fixed-budget comparison with ProMerge:
        the paper keeps every token receiving >=1 vote (variable count ~15%); here we
        keep the top-`keep_k` tokens ranked by vote count (with annealed Gumbel noise
        during training), so all variants share an identical token footprint. Gradients
        still flow through the soft per-token selection probability via the STE weight.
        Selection is HARD removal (not merging) — the deliberate contrast with ProMerge.
        """
        B, N, D = visual_tokens.shape
        Hv = visual_tokens

        # ---- Guidance tokens H_q = [H_l; H_p] ----
        guidance = [self._build_proprio_tokens(qpos)]  # H_p: [B, p, D]
        if slow_semantic is not None:
            Hl = self.tp_lang_proj(slow_semantic).unsqueeze(1)  # [B, 1, D]
            guidance.insert(0, Hl)
        Hq = torch.cat(guidance, dim=1)  # [B, Nq, D]

        Hv_n = rms_norm(Hv)
        Hq_n = rms_norm(Hq)

        # ---- Per-vision query and score matrix ----
        attn = torch.softmax(torch.bmm(Hv_n, Hq_n.transpose(1, 2)) / math.sqrt(D), dim=-1)  # [B, N, Nq]
        Q = torch.bmm(attn, Hq)  # [B, N, D]
        Qn = rms_norm(Q)
        S = torch.bmm(Qn, Hv_n.transpose(1, 2))  # [B, N, N]

        # ---- Annealed Gumbel perturbation (train only) ----
        if self.training:
            alpha = self._current_gumbel_alpha()
            U = torch.rand_like(S).clamp_(1e-9, 1.0)
            G = -torch.log(-torch.log(U))
            S_hat = S + alpha * G
            self.tp_step += 1
        else:
            alpha = self.tp_alpha_end
            S_hat = S

        # ---- Votes (argmax per row) and soft selection prob ----
        votes = S_hat.argmax(dim=-1)  # [B, N]
        vote_count = torch.zeros(B, N, device=Hv.device).scatter_add_(
            1, votes, torch.ones(B, N, device=Hv.device)
        )  # [B, N]
        P = torch.softmax(S_hat / alpha, dim=-1)  # [B, N, N]
        p_bar = P.mean(dim=1)  # [B, N] expected per-token selection probability

        # ---- Fixed-budget keep ranked by votes (tie-break by p_bar) ----
        rank = vote_count + p_bar
        topk_indices = torch.topk(rank, keep_k, dim=1).indices  # [B, keep_k]
        m = torch.zeros(B, N, device=Hv.device).scatter_(1, topk_indices, 1.0)

        # ---- Straight-through estimator: forward=hard, backward=soft ----
        w = m + p_bar - p_bar.detach()  # [B, N]
        Hv_cond = Hv * w.unsqueeze(-1)

        kept = torch.gather(Hv_cond, 1, topk_indices.unsqueeze(-1).expand(-1, -1, D))  # [B, keep_k, D]
        ctx = Hv.mean(dim=1, keepdim=True)  # [B, 1, D] global context token
        result_visual = torch.cat([kept, ctx], dim=1)  # [B, keep_k + 1, D]

        if pos_tokens is not None:
            C_pos = pos_tokens.shape[-1]
            if pos_tokens.shape[0] == 1:
                pos_tokens = pos_tokens.expand(B, -1, -1)
            pos_kept = torch.gather(pos_tokens, 1, topk_indices.unsqueeze(-1).expand(-1, -1, C_pos))
            pos_ctx = pos_tokens.mean(dim=1, keepdim=True)
            pos_out = torch.cat([pos_kept, pos_ctx], dim=1)
            return result_visual, pos_out
        return result_visual
