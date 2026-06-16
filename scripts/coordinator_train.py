import os
import sys
import subprocess
import re
import traceback

# Policy variants to evaluate
VARIANTS = [
    "PROMERGE_ONLY",
    "PROMERGE_FILM",
    "THINKPROPRIO"
]

# Each variant is trained through its self-contained baselines/<name>/train.py entry,
# which applies that baseline's backbone/hidden-dim config (single source of truth).
VARIANT_TO_FOLDER = {
    "MONOLITHIC_ACT": "baselines/monolithic_act/train.py",
    "RANDOM_PRUNE": "baselines/random_prune/train.py",
    "TOME_CLUSTERING": "baselines/tome_clustering/train.py",
    "PROMERGE_ONLY": "baselines/promerge_only/train.py",
    "PROMERGE_FILM": "baselines/promerge_film/train.py",
    "THINKPROPRIO": "baselines/thinkproprio/train.py",
}

def set_config_variant(variant_name):
    config_path = "src/config.py"
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace the variant field in CONFIG
    pattern = r'"variant":\s*PolicyVariant\.[A-Z_]+'
    replacement = f'"variant": PolicyVariant.{variant_name}'
    modified_content, count = re.subn(pattern, replacement, content)
    
    if count == 0:
        raise ValueError(f"Could not locate 'variant': PolicyVariant.<NAME> in config.py")
        
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(modified_content)
    print(f"🔄 Config updated variant to: {variant_name}")

def log_error(variant_name, error_msg):
    with open("error_log.txt", "a", encoding="utf-8") as f:
        f.write(f"=== Error in variant {variant_name} ===\n")
        f.write(error_msg + "\n\n")

def run_pipeline():
    python_bin = ".venv/bin/python"
    
    for variant in VARIANTS:
        print("\n" + "="*70)
        print(f"🏃 Starting training pipeline for: {variant}")
        print("="*70)
        
        # Step B: Checkpoint and log directories
        ckpt_dir = f"checkpoints/{variant}"
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Check if checkpoint and logs indicate at least 15 epochs have been completed
        ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
        log_path = os.path.join(ckpt_dir, "loss_log.txt")
        if os.path.exists(ckpt_path) and os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f_log:
                    lines = f_log.readlines()
                valid_lines = [l for l in lines if l.strip().startswith("Epoch")]
                if len(valid_lines) >= 50:
                    print(f"⏭️ Checkpoint for {variant} already exists with {len(valid_lines)} epochs. Skipping to next variant.")
                    continue
            except Exception as e:
                print(f"⚠️ Error checking checkpoint: {e}. Re-running training.")
        
        # Step A: Resolve this variant's self-contained training entrypoint.
        # (The folder's train.py applies the baseline's config, so we no longer
        #  mutate src/config.py here.)
        train_entry = VARIANT_TO_FOLDER.get(variant)
        if train_entry is None:
            print(f"❌ No baselines/ folder mapped for variant {variant}; skipping.")
            log_error(variant, f"No VARIANT_TO_FOLDER entry for {variant}")
            continue

        # Step C: Run training with batch size fallback logic (32 -> 16)
        batch_size = 32
        success = False

        while batch_size >= 16:
            print(f"👉 Launching {train_entry} with Batch Size: {batch_size}...")

            cmd = [
                python_bin,
                "-u",
                train_entry,
                "--epochs", "50",
                "--batch_size", str(batch_size),
            ]
            
            # Step D: Execute process and stream stdout in real-time
            try:
                # We use Popen to stream outputs to terminal
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True, 
                    bufsize=1
                )
                
                # Stream logs line-by-line
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    
                process.wait()
                exit_code = process.returncode
                
                if exit_code == 0:
                    print(f"✅ Training completed successfully for variant {variant} with batch size {batch_size}.")
                    success = True
                    break
                elif exit_code == 137:
                    print(f"⚠️ OOM/Device Allocation failure (Exit Code 137) detected at Batch Size {batch_size}.")
                    batch_size //= 2
                    if batch_size >= 16:
                        print(f"🔄 Retrying with smaller Batch Size {batch_size}...")
                    else:
                        print("❌ Cannot reduce batch size further. Skipping to next variant.")
                else:
                    print(f"❌ Training failed for variant {variant} with exit code {exit_code}.")
                    # Gather any remaining error outputs and log
                    log_error(variant, f"Training process exited with code {exit_code}")
                    break
                    
            except Exception as e:
                err_trace = traceback.format_exc()
                print(f"❌ Exception occurred: {e}")
                log_error(variant, err_trace)
                break
                
        if not success and batch_size < 16:
            log_error(variant, f"OOM failure: could not run training even with batch size 16")

if __name__ == "__main__":
    run_pipeline()
