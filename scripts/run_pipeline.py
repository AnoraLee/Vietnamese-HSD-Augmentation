import subprocess
import sys
import os

def run_script(script_path):
    print(f"\n{'='*60}")
    print(f"Running: {script_path}")
    print(f"{'='*60}\n")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, script_path)
    
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        return False
    
    result = subprocess.run(
        [sys.executable, full_path],
        cwd=base_dir,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"\nError in {script_path}:")
        print(result.stderr)
        return False
    
    print(f"\n{script_path} completed successfully!")
    return True

def main():
    print("\nVietnamese Hate Speech Detection - Pipeline")
    print("="*60)
    
    scripts = [
        "scripts/01_prepare_data.py",
        "scripts/02_eda_augmentation.py",
        # "scripts/03_back_translation.py",  # Comment lại
        # "scripts/04_train.py",
        # "scripts/05_evaluate.py",
    ]
    
    for script in scripts:
        if not run_script(script):
            print(f"\nPipeline failed at {script}")
            return 1
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    return 0

if __name__ == "__main__":
    sys.exit(main())