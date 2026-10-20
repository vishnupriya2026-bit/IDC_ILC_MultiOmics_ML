import os
import glob
import pandas as pd

project_dir = os.getcwd()
print("=" * 60)
print(f" PROJECT WORK SUMMARY: {os.path.basename(project_dir)}")
print("=" * 60)

# 1. Pipeline Scripts
print("\n[1] PIPELINE SCRIPTS (.py)")
py_files = sorted(glob.glob("*.py"))
for f in py_files:
    size_kb = os.path.getsize(f) / 1024
    print(f"  • {f:<35} ({size_kb:.1f} KB)")

# 2. Datasets & Input Data
print("\n[2] DATASETS & INPUT FILES")
data_files = glob.glob("data/**/*", recursive=True) + glob.glob("datasets/**/*", recursive=True)
data_files = [f for f in sorted(data_files) if os.path.isfile(f)]
for f in data_files:
    size_mb = os.path.getsize(f) / (1024 * 1024)
    if size_mb < 1.0:
        size_str = f"{os.path.getsize(f) / 1024:.1f} KB"
    else:
        size_str = f"{size_mb:.1f} MB"
    print(f"  • {f:<50} ({size_str})")

# 3. Generated Results & Figures
print("\n[3] GENERATED RESULTS & FIGURES (results/)")
results_files = sorted(glob.glob("results/*"))
for f in results_files:
    size_str = f"{os.path.getsize(f) / 1024:.1f} KB"
    print(f"  • {f:<50} ({size_str})")

# 4. Saved Models
print("\n[4] TRAINED ML MODELS (models/)")
model_files = sorted(glob.glob("models/*"))
for f in model_files:
    size_str = f"{os.path.getsize(f) / 1024:.1f} KB"
    print(f"  • {f:<50} ({size_str})")

print("\n" + "=" * 60)

# Write output summary to a text file for quick viewing/sharing
summary_txt_path = "results/project_directory_inventory.txt"
os.makedirs("results", exist_ok=True)
with open(summary_txt_path, "w") as out:
    out.write(f"Project Directory Inventory for {os.path.basename(project_dir)}\n")
    out.write("=" * 60 + "\n\n")
    
    out.write("Python Scripts:\n")
    for f in py_files: out.write(f" - {f}\n")
    
    out.write("\nData Files:\n")
    for f in data_files: out.write(f" - {f}\n")
    
    out.write("\nResults & Figures:\n")
    for f in results_files: out.write(f" - {f}\n")
    
    out.write("\nSaved Models:\n")
    for f in model_files: out.write(f" - {f}\n")

print(f"Saved full directory inventory to '{summary_txt_path}'")
