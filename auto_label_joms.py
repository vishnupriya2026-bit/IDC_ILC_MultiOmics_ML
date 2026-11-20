import os, glob, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BACKUP_DIR = "figures_unlabeled_backup"
if not os.path.exists(BACKUP_DIR):
    shutil.copytree("figures", BACKUP_DIR)
    print(f"Backed up original figures to {BACKUP_DIR}/\n")

os.makedirs("joms_submission_figures", exist_ok=True)
os.makedirs("preview_jpgs", exist_ok=True)

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

def find_intervals(is_white_1d, min_gap=50, edge_pad=15):
    """Find content intervals separated by interior white-space gutters."""
    non_white = np.where(~is_white_1d)[0]
    if len(non_white) == 0:
        return [(0, len(is_white_1d))]
    start, end = int(non_white[0]), int(non_white[-1])

    # Find contiguous white runs inside [start, end]
    sub = is_white_1d[start:end + 1]
    splits = []
    run_start = None
    for i, val in enumerate(sub):
        if val and run_start is None:
            run_start = i
        elif not val and run_start is not None:
            if i - run_start >= min_gap:
                splits.append((start + run_start, start + i))
            run_start = None

    intervals = []
    cur = max(0, start - edge_pad)
    for g_start, g_end in splits:
        mid = (g_start + g_end) // 2
        intervals.append((cur, mid))
        cur = mid + 10
    intervals.append((cur, min(len(is_white_1d), end + edge_pad)))
    return intervals

files = sorted(glob.glob(f"{BACKUP_DIR}/*.tiff"))
all_frames = []

print(f"{'No.':<4} {'Figure File':<52} {'Grid':<7} {'Labels Applied'}")
print("-" * 82)

for idx, src_path in enumerate(files, 1):
    fname = os.path.basename(src_path)
    im = Image.open(src_path).convert("RGB")
    gray = np.array(im.convert("L"))

    # Locate outer content bounding box
    mask = gray < 248
    ys, xs = np.where(mask)
    y0, y1 = (int(ys.min()), int(ys.max()) + 1) if len(ys) else (0, gray.shape[0])
    x0, x1 = (int(xs.min()), int(xs.max()) + 1) if len(xs) else (0, gray.shape[1])

    crop = gray[y0:y1, x0:x1]
    col_white = (crop >= 248).mean(axis=0) >= 0.993
    row_white = (crop >= 248).mean(axis=1) >= 0.993

    col_bins = [(x0 + a, x0 + b) for a, b in find_intervals(col_white, min_gap=55)]
    row_bins = [(y0 + a, y0 + b) for a, b in find_intervals(row_white, min_gap=55)]

    n_rows, n_cols = len(row_bins), len(col_bins)
    total_panels = n_rows * n_cols
    draw = ImageDraw.Draw(im)

    applied = []
    if total_panels > 1:
        p_idx = 0
        for r_start, _ in row_bins:
            for c_start, _ in col_bins:
                letter = chr(ord("a") + p_idx)
                lx = max(15, c_start)
                ly = max(15, r_start - 10)
                # White backing box ensures crisp legibility near plot borders
                bbox = draw.textbbox((lx, ly), letter, font=font)
                draw.rectangle([bbox[0] - 6, bbox[1] - 6, bbox[2] + 6, bbox[3] + 6], fill=(255, 255, 255))
                draw.text((lx, ly), letter, fill=(0, 0, 0), font=font)
                applied.append(letter)
                p_idx += 1

    print(f"{idx:02d}.  {fname:<52} {n_rows}x{n_cols:<5} {', '.join(applied) if applied else 'Single panel (no label)'}")

    # 1. Overwrite 600-DPI RGB LZW TIFF in figures/
    im.save(os.path.join("figures", fname), dpi=(600, 600), compression="tiff_lzw")

    # 2. Save sequential JoMS TIFF (Fig1.tif..Fig11.tif, FigS1.tif..FigS3.tif)
    joms_name = f"Fig{idx}.tif" if idx <= 11 else f"FigS{idx - 11}.tif"
    im.save(os.path.join("joms_submission_figures", joms_name), dpi=(600, 600), compression="tiff_lzw")
    all_frames.append(im)

    # 3. Save standard small JPEG preview that opens natively in VS Code
    pw = 1200
    ph = int(im.height * (pw / im.width))
    im.resize((pw, ph), Image.Resampling.LANCZOS).save(
        os.path.join("preview_jpgs", f"{idx:02d}_{joms_name.replace('.tif', '.jpg')}"),
        quality=90,
    )

# Rebuild 14-page 600-DPI TIFF bundle
all_frames[0].save(
    "JOMS_All_14_Figures_Multipage_600DPI.tiff",
    save_all=True,
    append_images=all_frames[1:],
    dpi=(600, 600),
    compression="tiff_lzw",
)
print("\nDone! Open the 'preview_jpgs/' folder in VS Code Explorer to inspect each labeled figure.")
