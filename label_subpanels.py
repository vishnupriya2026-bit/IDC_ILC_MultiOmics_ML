import os, glob, math, shutil
from PIL import Image, ImageDraw, ImageFont

# ==============================================================================
# EDIT COORDINATES BELOW AFTER CHECKING panel_coordinate_guide.png
# Format: ("label", x_px, y_px, white_background_badge)
# Leave single-panel figures commented out (#).
# ==============================================================================
PANEL_LABELS = {
    # 4500 x 1500 px -> 3 side-by-side panels (1500 px wide each)
    "Main_Fig01_DIABLO_Subspace_Alignment.tiff": [
        ("a", 35, 25, False),
        ("b", 1535, 25, False),
        ("c", 3035, 25, False),
    ],

    # Uncomment if Main_Fig03 (2700 x 1500 px) has 2 panels:
    # "Main_Fig03_Bayesian_Tuned_Model_Comparison.tiff": [
    #     ("a", 35, 25, False),
    #     ("b", 1385, 25, False),
    # ],

    # Uncomment if Main_Fig05 (2400 x 1500 px) has 2 panels:
    # "Main_Fig05_Decision_Threshold_Calibration.tiff": [
    #     ("a", 35, 25, False),
    #     ("b", 1235, 25, False),
    # ],

    # 4142 x 1530 px -> 3 panels (~1380 px each); if 4 panels, use x = 35, 1070, 2105, 3140
    "Main_Fig09_SHAP_Dependence_Plots.tiff": [
        ("a", 35, 25, False),
        ("b", 1415, 25, False),
        ("c", 2795, 25, False),
    ],

    # Uncomment if Main_Fig11 (2671 x 1470 px) has 2 panels:
    # "Main_Fig11_Kaplan_Meier_Survival_METABRIC.tiff": [
    #     ("a", 35, 25, False),
    #     ("b", 1370, 25, False),
    # ],

    # 4800 x 3000 px -> 2x2 grid (if 2x3 grid, use x = 35, 1635, 3235 on rows y = 25 and y = 1525)
    "Supplementary_Fig_S1_Preprocessing_QC_Benchmark.tiff": [
        ("a", 35, 25, False),
        ("b", 2435, 25, False),
        ("c", 35, 1525, False),
        ("d", 2435, 1525, False),
    ],

    # 3000 x 2400 px -> 2x2 grid of confusion matrices (1500 x 1200 px per quadrant)
    "Supplementary_Fig_S2_Holdout_Confusion_Matrices.tiff": [
        ("a", 35, 25, False),
        ("b", 1535, 25, False),
        ("c", 35, 1225, False),
        ("d", 1535, 1225, False),
    ],

    # 4200 x 3000 px -> 2x2 grid (if 2x3 grid, use x = 35, 1435, 2835 on rows y = 25 and y = 1525)
    "Supplementary_Fig_S3_OFAT_Sensitivity_Analysis.tiff": [
        ("a", 35, 25, False),
        ("b", 2135, 25, False),
        ("c", 35, 1525, False),
        ("d", 2135, 1525, False),
    ],
}

# Create one-time clean backup so re-running never double-stamps labels
BACKUP_DIR = "figures_unlabeled_backup"
if not os.path.exists(BACKUP_DIR):
    shutil.copytree("figures", BACKUP_DIR)
    print(f"Created clean backup in {BACKUP_DIR}/")

# JoMS standard: 11 pt bold sans-serif at 600 DPI (~92 px)
FONT_SIZE_PX = 92
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
font = next((ImageFont.truetype(fp, FONT_SIZE_PX) for fp in FONT_CANDIDATES if os.path.exists(fp)), ImageFont.load_default())

os.makedirs("joms_submission_figures", exist_ok=True)
files = sorted(glob.glob(f"{BACKUP_DIR}/*.tiff"))
all_frames = []
preview_thumbs = []

for idx, src_path in enumerate(files, 1):
    fname = os.path.basename(src_path)
    dst_path = os.path.join("figures", fname)
    im = Image.open(src_path).convert("RGB")
    draw = ImageDraw.Draw(im)

    if fname in PANEL_LABELS:
        for label, x, y, badge in PANEL_LABELS[fname]:
            if badge:
                bbox = draw.textbbox((x, y), label, font=font)
                draw.rectangle([bbox[0] - 12, bbox[1] - 12, bbox[2] + 12, bbox[3] + 12], fill=(255, 255, 255))
            draw.text((x, y), label, fill=(0, 0, 0), font=font)
        print(f"[LABELED] {fname} -> {[p[0] for p in PANEL_LABELS[fname]]}")
    else:
        print(f"[SINGLE ] {fname}")

    # Save 600-DPI RGB LZW TIFF to both figures/ and joms_submission_figures/
    im.save(dst_path, dpi=(600, 600), compression="tiff_lzw")
    joms_name = f"Fig{idx}.tif" if idx <= 11 else f"FigS{idx - 11}.tif"
    im.save(os.path.join("joms_submission_figures", joms_name), dpi=(600, 600), compression="tiff_lzw")
    all_frames.append(im)

    # Thumbnail for verification sheet
    tw = 850
    th = int(im.height * (tw / im.width))
    preview_thumbs.append((f"{joms_name} | {fname}", im.resize((tw, th), Image.Resampling.LANCZOS)))

# Rebuild 14-page 600-DPI multi-page TIFF
all_frames[0].save(
    "JOMS_All_14_Figures_Multipage_600DPI.tiff",
    save_all=True,
    append_images=all_frames[1:],
    dpi=(600, 600),
    compression="tiff_lzw",
)

# Build verification grid image
cols = 2
rows = math.ceil(len(preview_thumbs) / cols)
max_h = max(t.height for _, t in preview_thumbs) + 40
sheet = Image.new("RGB", (cols * 870 + 20, rows * (max_h + 20) + 20), (240, 240, 240))
pfont = ImageFont.load_default()

for i, (title, t_im) in enumerate(preview_thumbs):
    r, c = divmod(i, cols)
    x0, y0 = c * 870 + 10, r * (max_h + 20) + 10
    ImageDraw.Draw(sheet).rectangle([x0, y0, x0 + 850, y0 + 32], fill=(30, 30, 30))
    ImageDraw.Draw(sheet).text((x0 + 10, y0 + 8), title, fill=(255, 255, 255), font=pfont)
    sheet.paste(t_im, (x0, y0 + 32))

sheet.save("labeled_preview_sheet.png")
print("\nDone! Inspect labeled_preview_sheet.png in VS Code Explorer to verify all label positions.")
