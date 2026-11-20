import os, glob, shutil
from PIL import Image, ImageDraw, ImageFont

BACKUP_DIR = "figures_unlabeled_backup"

# Set (rows, cols) for each multi-panel figure; set None for single-panel figures
GRID_LAYOUT = {
    "Main_Fig01_DIABLO_Subspace_Alignment.tiff":            (1, 3),  # a, b, c
    "Main_Fig02_Circos_Correlation_Map.tiff":               None,    # Single panel
    "Main_Fig03_Bayesian_Tuned_Model_Comparison.tiff":      None,    # Change to (1, 2) if 2 panels
    "Main_Fig04_10Fold_CV_ROC_Curves.tiff":                 None,    # Single panel
    "Main_Fig05_Decision_Threshold_Calibration.tiff":       None,    # Change to (1, 2) if 2 panels
    "Main_Fig06_Biomarker_Panel_Parsimony.tiff":            None,    # Single panel
    "Main_Fig07_SHAP_Global_Feature_Importance.tiff":       None,    # Change to (1, 2) if 2 panels
    "Main_Fig08_SHAP_Beeswarm_Summary.tiff":                None,    # Single panel
    "Main_Fig09_SHAP_Dependence_Plots.tiff":                (1, 3),  # a, b, c
    "Main_Fig10_Decision_Curve_Analysis_METABRIC.tiff":     None,    # Single panel
    "Main_Fig11_Kaplan_Meier_Survival_METABRIC.tiff":       None,    # Single panel
    "Supplementary_Fig_S1_Preprocessing_QC_Benchmark.tiff": (2, 2),  # Change to (2, 3) if 6 panels
    "Supplementary_Fig_S2_Holdout_Confusion_Matrices.tiff": (2, 2),  # a, b, c, d
    "Supplementary_Fig_S3_OFAT_Sensitivity_Analysis.tiff":  (2, 2),  # Change to (2, 3) if 6 panels
}

shutil.rmtree("joms_submission_figures", ignore_errors=True)
shutil.rmtree("preview_pngs", ignore_errors=True)
os.makedirs("joms_submission_figures", exist_ok=True)
os.makedirs("preview_pngs", exist_ok=True)

# JoMS standard: 11 pt bold sans-serif at 600 DPI (~92 px)
FONT_SIZE_PX = 92
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
font = next(
    (ImageFont.truetype(fp, FONT_SIZE_PX) for fp in FONT_CANDIDATES if os.path.exists(fp)),
    ImageFont.load_default(),
)

files = sorted(glob.glob(f"{BACKUP_DIR}/*.tiff"))
print(f"{'No.':<4} {'JoMS File':<12} {'Grid':<7} {'Labels Applied'}")
print("-" * 55)

for idx, src_path in enumerate(files, 1):
    fname = os.path.basename(src_path)
    im = Image.open(src_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)

    grid = GRID_LAYOUT.get(fname)
    applied = []

    if grid and grid[0] * grid[1] > 1:
        rows, cols = grid
        cell_w = w / cols
        cell_h = h / rows
        p_idx = 0
        for r in range(rows):
            for c in range(cols):
                letter = chr(ord("a") + p_idx)
                lx = int(c * cell_w + 30)
                ly = int(r * cell_h + 25)
                bbox = draw.textbbox((lx, ly), letter, font=font)
                draw.rectangle([bbox[0] - 6, bbox[1] - 6, bbox[2] + 6, bbox[3] + 6], fill=(255, 255, 255))
                draw.text((lx, ly), letter, fill=(0, 0, 0), font=font)
                applied.append(letter)
                p_idx += 1

    joms_name = f"Fig{idx}.tif" if idx <= 11 else f"FigS{idx - 11}.tif"

    # Save 600-DPI 8-bit RGB LZW TIFFs with zero titles or extra text
    im.save(os.path.join("figures", fname), dpi=(600, 600), compression="tiff_lzw")
    im.save(os.path.join("joms_submission_figures", joms_name), dpi=(600, 600), compression="tiff_lzw")

    # Save PNG preview for VS Code Explorer
    pw = 1100
    ph = int(h * (pw / w))
    im.resize((pw, ph), Image.Resampling.LANCZOS).save(
        os.path.join("preview_pngs", joms_name.replace(".tif", ".png"))
    )

    grid_str = f"{grid[0]}x{grid[1]}" if grid else "1x1"
    label_str = ", ".join(applied) if applied else "None (single panel)"
    print(f"{idx:02d}.  {joms_name:<12} {grid_str:<7} {label_str}")

shutil.make_archive("JOMS_Submission_Figures", "zip", "joms_submission_figures")
print("\nUpdated JOMS_Submission_Figures.zip and preview_pngs/.")
