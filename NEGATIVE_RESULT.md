# ProMerge: A Negative Result on Front-End Visual Token Compression for Lightweight (no-VLM) Manipulation Policies

**Status: idea did not pan out. Documented honestly as a negative result.**
2026-06-18

## TL;DR

We investigated whether **intent-guided visual token compression** in a
**lightweight, VLM-free** Vision-Action backbone (ViT-Small + ACT) could match
or beat heavier VLA pipelines on LIBERO manipulation, while running at high
frequency. The answer, after systematic experiments, is **no**:

- Across 3 LIBERO suites and 3 random seeds, our best configuration averages
  **~60-67%** success and **never reliably beats** a parameter-free pure-saliency
  token merger (ToMe, **73%**), let alone full-token methods (CT-VAM, **82%**).
- The core issue is not implementation tuning — it is the **idea itself**:
  in a no-VLM backbone, front-end token compression *removes information without
  a reliable grounding signal*, and the "speed" it buys is paid for in success
  rate, which is the wrong trade in manipulation.

## What we tried (and the numbers)

Backbone ViT-Small + ACT, LIBERO, 20 episodes/task, full 10-task eval.

| Method / config | Success |
|---|---|
| Pure saliency merge (ToMe) | **73.0%** |
| monolithic (no compression) | 61.0% |
| **promerge_film** (intent gate, dot, sum-saliency) | 66.5% (single) |
| promerge_film, cross-attention intent | 64.5% |
| promerge_film, CLIP-vision backbone | 44.5% |
| promerge_film, **fixed saliency** (uniqueness g_vis) | 75.5% single → **60.2% (3-seed avg, 53/64/63.5)** |
| + small-init (variance fix) | 67.5 / 64.0 (variance ↓, mean still ~65) |
| cross-suite (object / goal) | 61.0% / 69.5% |

Reference (from literature, not our runs): ThinkProprio 97% and CT-VAM 82% are
both stronger, but ThinkProprio uses a Florence-2 VLM + DiT, and CT-VAM keeps
**all** visual tokens (no compression) and innovates in the *decoder*.

## Diagnostics that led to the conclusion (the useful part)

1. **Saliency gate was inverted.** The original `g_vis = self_attn.sum()`
   measured *background homogeneity* (big uniform regions score high), not
   saliency — visualizations showed it highlighting the large dark cabinet, not
   the small target bowl. Switching to a cosine-uniqueness saliency fixed the
   heatmap (target objects light up). This was a real bug; fixing it helped but
   did not close the gap to ToMe.
2. **High seed variance (53-75%)** traced to random init of the intent filter.
   Small-init reduced variance (64-67.5) but did not raise the ceiling.
3. **Intent grounding never localized the target.** Probing g_kin across
   instructions showed weak, non-target-specific responses — the CLIP-text vs.
   ImageNet-ViT (and even CLIP-ViT) features are not aligned enough for a
   no-VLM dot-product / cross-attention to ground language to objects.

## Why the idea is flawed (the real lesson)

- **Token compression is not the cerebellum's job.** Deciding *where to look*
  (visual selection) is cortical top-down attention + collicular saliency, not
  low-level visuomotor execution. Putting compression inside a "lightweight
  cerebellum" is biologically inconsistent — and empirically, the methods that
  work in the no-VLM regime (CT-VAM, VITA) **do not compress** front-end tokens.
- **The value proposition ("fast") optimizes the wrong axis.** In manipulation,
  success rate is the binding constraint; speed is secondary. Our compression
  trades success for speed, and the trade is unfavorable.
- **No-VLM + token compression is the empty-but-empty cell.** Every token-pruning
  VLA (LightVLA/ThinkProprio/VLA-Pruner) is parasitic on a 7B VLM precisely
  because that is where compression pays off. Without a VLM there is no reliable
  grounding to select tokens by, so compression mostly destroys information.

## Conclusion

The negative result is robust (3 suites × 3 seeds, multiple grounding
mechanisms, a fixed saliency gate, and a variance fix all fail to beat a
parameter-free baseline). The right call is to stop, record this, and not
sink further cost. The artifacts here (suite-switchable harness, gate ablation
switches, attention visualizations, 3-seed protocol) remain a working,
reproducible experimental pipeline.
