import os
import glob

print("=" * 60)
print("              WORKSPACE & FIGURE CHECKER              ")
print("=" * 60)

cwd = os.getcwd()
print(f"\n[1] Current Working Directory:\n    {cwd}\n")

print("[2] Searching for all Figure Files (.png, .jpg, .svg, .pdf) in Workspace:")
figure_extensions = ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.pdf']
found_figures = []

for ext in figure_extensions:
    found_figures.extend(glob.glob(os.path.join(cwd, "**", ext), recursive=True))

if found_figures:
    for fig in found_figures:
        print(f"    - {fig}")
else:
    print("    No figure files found in current or subdirectories.")

print("\n[3] Key Directory Contents:")
for item in sorted(os.listdir(cwd)):
    item_path = os.path.join(cwd, item)
    kind = "Folder" if os.path.isdir(item_path) else "File"
    print(f"    - [{kind}] {item}")

print("=" * 60)
