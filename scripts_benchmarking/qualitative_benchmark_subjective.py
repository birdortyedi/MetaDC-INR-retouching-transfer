import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import subprocess
import yaml
import shutil
import time

# Configuration
DATA_DIR = "/home/birdortyedi/inr-retouching/data/Subjective_Evaluation_Data"
METHOD1_DIR = "/home/birdortyedi/PycharmProjects/InRetouch"
METHOD2_DIR = "/home/birdortyedi/Downloads/InRetouch_backup/InRetouch"
RESULTS_DIR = "/home/birdortyedi/inr-retouching/QUALITATIVE_RESULTS"
CONDA_ENV = "mauve"

# Get all samples automatically
SAMPLES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

os.makedirs(os.path.join(RESULTS_DIR, "M1"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "M2"), exist_ok=True)

def create_qual_yml(method_dir, sample_id, base_yml_name, name):
    base_path = os.path.join(method_dir, "options", "train", base_yml_name)
    with open(base_path, "r") as f:
        config = yaml.safe_load(f)
    
    sample_dir = os.path.join(DATA_DIR, sample_id)
    before_img = os.path.join(sample_dir, f"{sample_id}_before.jpg")
    after_img = os.path.join(sample_dir, f"{sample_id}_after.jpg")
    input_img_dir = sample_dir
    
    config['name'] = f"Q_{name}_{sample_id}"
    config['datasets']['train']['style_natural'] = [before_img]
    config['datasets']['train']['style_output'] = [after_img]
    config['datasets']['val']['inp_natural'] = input_img_dir
    config['val']['save_img'] = True
    
    # Save the YAML
    qual_path = os.path.join(method_dir, "options", "train", f"q_subj_{sample_id}.yml")
    with open(qual_path, "w") as f:
        yaml.dump(config, f)
    return qual_path

def run_and_extract(method_dir, sample_id, yml_path, method_name):
    # Check if already exists
    # We look for .jpg or .png
    final_dst_base = os.path.join(RESULTS_DIR, method_name, sample_id)
    if os.path.exists(final_dst_base + ".jpg") or os.path.exists(final_dst_base + ".png"):
        print(f"Skipping {method_name} for {sample_id} - already exists.")
        return

    print(f"\n>>> Running {method_name} for {sample_id}...")
    cmd = ["conda", "run", "-n", CONDA_ENV, "python", "basicsr/train_INR.py", "-opt", yml_path]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{method_dir}:{env.get('PYTHONPATH', '')}"
    
    # Run it
    subprocess.run(cmd, cwd=method_dir, env=env)
    
    # Extract the result
    vis_root = os.path.join(method_dir, "experiments", f"Q_{method_name}_{sample_id}", "visualization")
    
    found = False
    for root, _, files in os.walk(vis_root):
        for f in files:
            if "_input" in f and (f.endswith(".jpg") or f.endswith(".png")):
                src_path = os.path.join(root, f)
                ext = os.path.splitext(f)[1]
                dst_path = final_dst_base + ext
                shutil.copy(src_path, dst_path)
                print(f"✅ Saved to: {dst_path}")
                found = True
                break
        if found: break
    
    if not found:
        print(f"❌ Failed to extract result for {method_name} - {sample_id}")
    
    # Clean up experiment folder to save space
    exp_folder = os.path.join(method_dir, "experiments", f"Q_{method_name}_{sample_id}")
    if os.path.exists(exp_folder):
        shutil.rmtree(exp_folder)

if __name__ == "__main__":
    print(f"Starting full qualitative benchmark for {len(SAMPLES)} samples.")
    for sample_id in SAMPLES:
        # Method 1
        yml1 = create_qual_yml(METHOD1_DIR, sample_id, "InRetouch_Optimize_Single.yml", "M1")
        run_and_extract(METHOD1_DIR, sample_id, yml1, "M1")
        
        # Method 2
        yml2 = create_qual_yml(METHOD2_DIR, sample_id, "InRetouch_Optimize_Single.yml", "M2")
        run_and_extract(METHOD2_DIR, sample_id, yml2, "M2")

    print(f"\nALL DONE! All results are in: {RESULTS_DIR}")
