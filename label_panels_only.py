import os, glob, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BACKUP_DIR = "figures_unlabeled_backup"
if not os.path.exists(BACKUP_DIR):
    shutil.copytree("figures", BACKUP_DIR)

shutil.rmtree("joms_submission_figures", ignore_errors=True)
shutil.rmtree("preview_pngs", ignore_errors=True)
os.makedirs("joms_submission_figures", exist_ok=True)
os.makedirs("preview_pngs", exist_ok=True)

FONT_SIZE_PX = 92
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
font = next(
    (ImageFont.truetype(fp, FONT_SIZE_PX) for fp in FONT_CANDIDATES if os.path.exists(fp)),
    ImageFont.load_default(),
)

def find_intervals(is_white_1d, min_gap=55, edge_pad=12):
    non_white = np.where(~is_white_1d)[0]
    if len(non_white) == 0:
        return [(0, len(is_white_1d))]
    start, end = int(non_white[0]), int(non_white[-1])
    sub = is_white_1d[start:end + 1]
    splits, run_start = [], None
    for i, val in enumerate(sub):
        if val and run_start is None:
            run_start = i
        elif not val and run_start is not None:
            if i - run_start >= min_gap:
                splits.append((start + run_start, start + i))
            run_start = None
    intervals, cur = [], max(0, start - edge_pad)
    for g_start, g_end in splits:
        mid = (g_start + g_end) // 2
        intervals.append((cur, mid))
        cur = mid + 10
    intervals.append((cur, min(len(is_white_1d), end + edge_pad)))
    return intervals

files = sorted(glob.glob(f"{BACKUP_DIR}/*.tiff"))
print(f"{'No.':<4} {'JoMS File':<12} {'Original File':<52} {'Labels Added'}")
print("-" * 88)

for idx, src_path in enumerate(files, 1):
    fname = os.path.basename(src_path)
    im = Image.open(src_path).convert("RGB")
    w, h = im.size
    gray = np.array(im.convert("L"))

    ys, xs = np.where(gray < 248)
    y0, y1 = (int(ys.min()), int(ys.max()) + 1) if len(ys) else (0, h)
    x0, x1 = (int(xs.min()), int(xs.max()) + 1) if len(xs) else (0, w)
    crop = gray[y0:y1, x0:x1]

    col_white = (crop >= 248).mean(axis=0) >= 0.993
    row_white = (crop >= 248).mean(axis=1) >= 0.993

    col_bins = [(x0 + a, x0 + b) for a, b in find_intervals(col_white, min_gap=55)]
    row_bins = [(y0 + a, y0 + b) for a, b in find_intervals(row_white, min_gap=55)]
    n_rows, n_cols = len(row_bins), len(col_bins)

    draw = ImageDraw.Draw(im)
    applied = []

    if n_rows * n_cols > 1:
        p_idx = 0
        for r_start, _ in row_bins:
            for c_start, _ in col_bins:
                letter = chr(ord("a") + p_idx)
                lx = max(18, c_start)
                ly = max(18, r_start - 8)
                bbox = draw.textbbox((lx, ly), letter, font=font)
                draw.rectangle([bbox[0] - 6, bbox[1] - 6, bbox[2] + 6, bbox[3] + 6], fill=(255, 255, 255))
                draw.text((lx, ly), letter, fill=(0, 0, 0), font=font)
                applied.append(letter)
                p_idx += 1

    joms_name = f"Fig{idx}.tif" if idx <= 11 else f"FigS{idx - 11}.tif"

    im.save(os.path.join("figures", fname), dpi=(600, 600), compression="tiff_lzw")
    im.save(os.path.join("joms_submission_figures", joms_name), dpi=(600, 600), compression="tiff_lzw")

    pw = 1100
    ph = int(h * (pw / w))
    im.resize((pw, ph), Image.Resampling.LANCZOS).save(
        os.path.join("preview_pngs", joms_name.replace(".tif", ".png"))
    )

    label_summary = ", ".join(applied) if applied else "None (single panel)"
    print(f"{idx:02d}.  {joms_name:<12} {fname:<52} {label_summary}")

shutil.make_archive("JOMS_Submission_Figures", "zip", "joms_submission_figures")
print("\nDone! Check preview_pngs/ to inspect and download JOMS_Submission_Figures.zip.")
