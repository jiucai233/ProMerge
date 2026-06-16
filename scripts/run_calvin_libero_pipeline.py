import subprocess
import re
import os
import sys
import argparse

# Make src/ importable so we can read the real CONFIG (image size, keep ratio, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Scenarios / Variants configurations
VARIANTS_CONFIG = [
    ("MONOLITHIC_ACT", "True", "Native ACT (No compression)"),
    ("RANDOM_PRUNE", "False", "Random Pruning (30% remaining)"),
    ("TOME_CLUSTERING", "False", "Competitor: Hard Selection (ToMe-style topk)"),
    ("TOME_CLUSTERING", "True", "Competitor: Token Merging (ToMe clustering)"),
    ("PROMERGE_ONLY", "False", "ProMerge Only: Hard Selection (No FiLM, Ours)"),
    ("PROMERGE_ONLY", "True", "ProMerge Only: Token Merging (No FiLM, Ours)"),
    ("THINKPROPRIO", "False", "ThinkProprio (Paper Reimpl.): Vote-based Hard Pruning"),
    ("PROMERGE_FILM", "True", "ProMerge Final: Token Merging (VLA + ToMe, Ours)")
]

def compute_analytical_gflops(variant_name, merge_mode, num_cameras=2, keep_ratio=None):
    """Analytical inference FLOPs (FLOPs = 2 x MACs) that reflect the ACTUAL per-variant
    architecture as configured in src/config.py and sim/calvin_libero_benchmark.py.

    Key facts the previous version got wrong and this one respects:
      * MONOLITHIC_ACT / RANDOM_PRUNE / TOME_CLUSTERING run on ResNet18 + hidden_dim 512.
      * PROMERGE_ONLY / PROMERGE_FILM run on ViT-Small + hidden_dim 384, with mid-layer
        token pruning so the second half of the ViT runs on the reduced token set.
      * Everything runs at CONFIG['image_size'] (240x320), not 480x640.
      * Token counts come from the real backbone strides, not a hardcoded 600.
    """
    # Read the real config when possible; fall back to known defaults otherwise.
    try:
        from config import CONFIG
        H, W = CONFIG.get("image_size", (240, 320))
        keep_ratio = CONFIG.get("keep_ratio", 0.3) if keep_ratio is None else keep_ratio
        prune_layer = CONFIG.get("vit_pruning_layer", 6)
    except Exception:
        H, W = (240, 320)
        keep_ratio = keep_ratio if keep_ratio is not None else 0.3
        prune_layer = 6

    def attn_macs(Lq, Lkv, dim):
        # q/out proj (2*Lq) + k/v proj (2*Lkv), each *dim*dim, plus QK^T and A·V (2*Lq*Lkv*dim)
        return 2 * Lq * dim * dim + 2 * Lkv * dim * dim + 2 * Lq * Lkv * dim

    def mlp_macs(L, dim, ffn):
        return 2 * L * dim * ffn

    # ViT-Small backbone variants (ProMerge + the ThinkProprio reimplementation).
    is_promerge = variant_name in ("PROMERGE_ONLY", "PROMERGE_FILM", "THINKPROPRIO")

    # --- hidden dims for the ACT transformer (matches run_evaluation routing) ---
    if is_promerge:
        d, ff = 384, 1536
    else:
        d, ff = 512, 3200

    macs = 0.0

    # --- Visual backbone ---
    if is_promerge:
        patch = 16
        ppc = (H // patch) * (W // patch)          # patches per camera (15*20 = 300)
        n_full = ppc * num_cameras                  # tokens before pruning
        n_keep = int(n_full * keep_ratio)           # joint tokens kept after gate
        depth = 12                                  # ViT-Small blocks
        # patch-embed conv
        macs += n_full * (patch * patch * 3 * d)
        # first half: per-camera self-attention over ppc tokens
        macs += num_cameras * prune_layer * (attn_macs(ppc, ppc, d) + mlp_macs(ppc, d, 4 * d))
        # second half: runs jointly on the pruned token set
        macs += (depth - prune_layer) * (attn_macs(n_keep, n_keep, d) + mlp_macs(n_keep, d, 4 * d))
        # gatekeeper selector ~ O(N_v^2 * D) score matrix
        macs += n_full * n_full * d
        n_vis = n_keep
    else:
        gh = -(-H // 32)                            # ResNet18 stride 32, ceil-divide
        gw = -(-W // 32)
        n_full = gh * gw * num_cameras
        # ResNet18: 1.82 GMACs @224x224, scaled by pixel area and #cameras
        macs += 1.82e9 * (H * W) / (224 * 224) * num_cameras
        # ResNet baselines prune AFTER the backbone, so only the ACT sees fewer tokens.
        n_vis = n_full if variant_name == "MONOLITHIC_ACT" else int(n_full * keep_ratio)

    # --- ACT transformer at inference: encoder over memory + decoder for 100 queries ---
    enc_layers, dec_layers, n_queries = 4, 7, 100
    mem = n_vis + 2                                  # visual tokens + proprio + latent
    macs += enc_layers * (attn_macs(mem, mem, d) + mlp_macs(mem, d, ff))
    macs += dec_layers * (
        attn_macs(n_queries, n_queries, d)          # decoder self-attention
        + attn_macs(n_queries, mem, d)              # decoder cross-attention to memory
        + mlp_macs(n_queries, d, ff)
    )

    gflops = 2.0 * macs / 1e9                        # FLOPs = 2 x MACs
    return round(gflops, 2)

def _col_best(rows, key, lower_better=False):
    vals = [r.get(key, 0.0) for r in rows]
    if not vals:
        return None
    return min(vals) if lower_better else max(vals)


def _fmt(val, best, pct=True, dec=1):
    s = f"{val:.{dec}f}%" if pct else f"{val:.{dec}f}"
    # Bold the column-best cell ("best in bold", per the table caption).
    return f"**{s}**" if (best is not None and abs(val - best) < 1e-9) else s


def main():
    parser = argparse.ArgumentParser(description="CALVIN and LIBERO aggregation pipeline")
    parser.add_argument("--num_calvin_rollouts", type=int, default=10,
                        help="Number of evaluation rollouts for CALVIN.")
    parser.add_argument("--num_libero_rollouts", type=int, default=3,
                        help="Number of evaluation rollouts for each LIBERO task.")
    args = parser.parse_args()
    
    python_bin = ".venv/bin/python"
    project_root = "/Users/jiucai/my_codes/ProMerge"
    
    # Nested dictionary to store all parsed metrics
    # results[pretty_name] = { calvin_metrics, libero_metrics }
    results = {}
    
    for base_variant, merge_mode, pretty_name in VARIANTS_CONFIG:
        print(f"\n========================================================")
        print(f"🎬 Running Evaluation for Variant: {pretty_name} ({base_variant}, merge={merge_mode})")
        print(f"========================================================")
        
        cmd = [
            python_bin,
            "-u",
            "sim/calvin_libero_benchmark.py",
            "--variant", base_variant,
            "--num_calvin_rollouts", str(args.num_calvin_rollouts),
            "--num_libero_rollouts", str(args.num_libero_rollouts),
            "--merge_tokens", merge_mode
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=project_root
        )
        
        stdout_lines = []
        while True:
            line = process.stdout.readline()
            if not line:
                break
            sys.stdout.write(line)
            sys.stdout.flush()
            stdout_lines.append(line)
            
        process.wait()
        
        # Parse metrics output line
        # Format: METRICS_OUTPUT: name | lh1 | lh2 | lh3 | lh4 | lh5 | avg_len | spatial | object | goal | long | avg
        parsed = False
        for line in stdout_lines:
            if line.startswith("METRICS_OUTPUT:"):
                parts = line.strip().split("|")
                if len(parts) >= 12:
                    lh1 = float(parts[1])
                    lh2 = float(parts[2])
                    lh3 = float(parts[3])
                    lh4 = float(parts[4])
                    lh5 = float(parts[5])
                    avg_len = float(parts[6])
                    
                    spatial = float(parts[7])
                    obj = float(parts[8])
                    goal = float(parts[9])
                    long_lh = float(parts[10])
                    libero_avg = float(parts[11])
                    
                    results[pretty_name] = {
                        "lh1": lh1, "lh2": lh2, "lh3": lh3, "lh4": lh4, "lh5": lh5, "avg_len": avg_len,
                        "spatial": spatial, "object": obj, "goal": goal, "long": long_lh, "libero_avg": libero_avg,
                        "gflops": compute_analytical_gflops(base_variant, merge_mode)
                    }
                    parsed = True
                    break
                    
        if not parsed:
            print(f"⚠️ Warning: Could not parse metrics output for {pretty_name}. Setting defaults to 0.0")
            results[pretty_name] = {
                "lh1": 0.0, "lh2": 0.0, "lh3": 0.0, "lh4": 0.0, "lh5": 0.0, "avg_len": 0.0,
                "spatial": 0.0, "object": 0.0, "goal": 0.0, "long": 0.0, "libero_avg": 0.0,
                "gflops": compute_analytical_gflops(base_variant, merge_mode)
            }
            
    # Generate the Markdown matrix reports
    print("\n\nGenerating final reports...")
    
    report_lines = []
    report_lines.append("# ProMerge vs. ThinkProprio: CALVIN & LIBERO Benchmark Emulation Report\n")
    report_lines.append(f"This report presents the local physical simulation benchmark emulation results comparing **Token Merging (ProMerge, Ours)** against **Hard Selection (ThinkProprio style pruning)** and other baselines. Evaluated on the local 9-DOF MuJoCo Franka Emika Panda sorting sandbox (CALVIN: {args.num_calvin_rollouts} rollouts, LIBERO: {args.num_libero_rollouts} rollouts per task).\n")
    
    # 1. CALVIN ABC->D Table
    report_lines.append("## 📊 Table 2. Results on CALVIN ABC→D Emulation")
    report_lines.append("Success rate (%) at each chain length and average completed chain length (Avg. Len.). Best in bold.\n")
    report_lines.append("| Method | GFLOPs ↓ | LH-1 ↑ | LH-2 ↑ | LH-3 ↑ | LH-4 ↑ | LH-5 ↑ | Avg. Len. ↑ |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    calvin_rows = [results.get(pn, {}) for _, _, pn in VARIANTS_CONFIG]
    cbest = {
        "gflops": _col_best(calvin_rows, "gflops", lower_better=True),
        "lh1": _col_best(calvin_rows, "lh1"), "lh2": _col_best(calvin_rows, "lh2"),
        "lh3": _col_best(calvin_rows, "lh3"), "lh4": _col_best(calvin_rows, "lh4"),
        "lh5": _col_best(calvin_rows, "lh5"), "avg_len": _col_best(calvin_rows, "avg_len"),
    }
    for _, _, pretty_name in VARIANTS_CONFIG:
        m = results.get(pretty_name, {})
        row = (f"| `{pretty_name}` "
               f"| {_fmt(m.get('gflops',0.0), cbest['gflops'], pct=False, dec=2)} "
               f"| {_fmt(m.get('lh1',0.0), cbest['lh1'])} "
               f"| {_fmt(m.get('lh2',0.0), cbest['lh2'])} "
               f"| {_fmt(m.get('lh3',0.0), cbest['lh3'])} "
               f"| {_fmt(m.get('lh4',0.0), cbest['lh4'])} "
               f"| {_fmt(m.get('lh5',0.0), cbest['lh5'])} "
               f"| {_fmt(m.get('avg_len',0.0), cbest['avg_len'], pct=False, dec=2)} |")
        report_lines.append(row)
        
    report_lines.append("\n\n")
    
    # 2. LIBERO Suites Table
    report_lines.append("## 📊 Table 3. Results on LIBERO Benchmark Suites Emulation")
    report_lines.append("Average success rate (%) across 10 tasks per suite. Best is in bold.\n")
    report_lines.append("| Method | GFLOPs ↓ | Spatial ↑ | Object ↑ | Goal ↑ | Long ↑ | Avg. ↑ |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    libero_rows = [results.get(pn, {}) for _, _, pn in VARIANTS_CONFIG]
    lbest = {
        "gflops": _col_best(libero_rows, "gflops", lower_better=True),
        "spatial": _col_best(libero_rows, "spatial"), "object": _col_best(libero_rows, "object"),
        "goal": _col_best(libero_rows, "goal"), "long": _col_best(libero_rows, "long"),
        "libero_avg": _col_best(libero_rows, "libero_avg"),
    }
    for _, _, pretty_name in VARIANTS_CONFIG:
        m = results.get(pretty_name, {})
        row = (f"| `{pretty_name}` "
               f"| {_fmt(m.get('gflops',0.0), lbest['gflops'], pct=False, dec=2)} "
               f"| {_fmt(m.get('spatial',0.0), lbest['spatial'])} "
               f"| {_fmt(m.get('object',0.0), lbest['object'])} "
               f"| {_fmt(m.get('goal',0.0), lbest['goal'])} "
               f"| {_fmt(m.get('long',0.0), lbest['long'])} "
               f"| {_fmt(m.get('libero_avg',0.0), lbest['libero_avg'])} |")
        report_lines.append(row)
        
    report_str = "\n".join(report_lines)
    
    os.makedirs("result", exist_ok=True)
    report_path = "result/calvin_libero_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)
        
    print(f"\n✨ Compiled reports written successfully to {report_path}!")

if __name__ == "__main__":
    main()
