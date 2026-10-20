import torch
import subprocess

def check_setup():
    print("=== GPU & CUDA Verification ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
    else:
        print("WARNING: CUDA is not detected by PyTorch.")

    print("\n=== R & mixOmics Verification ===")
    try:
        r_check = subprocess.run(
            ["Rscript", "-e", "library(mixOmics); cat('mixOmics version:', as.character(packageVersion('mixOmics')), '\\n')"],
            capture_output=True, text=True, check=True
        )
        print(r_check.stdout.strip())
    except Exception as e:
        print(f"R/mixOmics Check Failed: {e}")

if __name__ == "__main__":
    check_setup()
