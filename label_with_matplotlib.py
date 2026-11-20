import os, io, glob, shutil, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BACKUP_DIR = "figures_unlabeled_backup"
if not os.path.exists(BACKUP_DIR):
    shutil.copytree("figures", BACKUP_DIR)
    print(f"Backed up original unlabeled TIFFs to {BACKUP_DIR}/\n")

os.makedirs("joms_submission_figures", exist_ok=True)
os.makedirs("preview_pngs", exist_ok=True)

def find_intervals(is_white_1d, min_gap=60, edge_pad=12):
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
all_frames = []
html_cards = []

print(f"{'No.':<4} {'Figure Filename':<52} {'Layout':<7} {'JoMS Labels'}")
print("-" * 82)

for idx, src_path in enumerate(files, 1):
    fname = os.path.basename(src_path)
    im = Image.open(src_path).convert("RGB")
    w, h = im.size
    arr = np.array(im)
    gray = np.array(im.convert("L"))

    # Detect outer plot bounds and interior white gutters
    ys, xs = np.where(gray < 248)
    y0, y1 = (int(ys.min()), int(ys.max()) + 1) if len(ys) else (0, h)
    x0, x1 = (int(xs.min()), int(xs.max()) + 1) if len(xs) else (0, w)
    crop = gray[y0:y1, x0:x1]

    # Ignore top 8% when scanning vertical gutters in case of a shared super-title
    top_trim = int(0.08 * crop.shape[0])
    col_white = (crop[top_trim:, :] >= 248).mean(axis=0) >= 0.993
    row_white = (crop >= 248).mean(axis=1) >= 0.993

    col_bins = [(x0 + a, x0 + b) for a, b in find_intervals(col_white, min_gap=60)]
    row_bins = [(y0 + a, y0 + b) for a, b in find_intervals(row_white, min_gap=60)]
    n_rows, n_cols = len(row_bins), len(col_bins)

    # Render at exact native pixel dimensions & 600 DPI using Matplotlib
    dpi = 600
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr, aspect="auto")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    applied = []
    if n_rows * n_cols > 1:
        p_idx = 0
        for r_start, _ in row_bins:
            for c_start, _ in col_bins:
                letter = chr(ord("a") + p_idx)
                lx = max(18, c_start)
                ly = max(18, r_start)
                ax.text(
                    lx, ly, letter,
                    fontsize=11,
                    fontweight="bold",
                    fontfamily="DejaVu Sans",
                    color="black",
                    va="top", ha="left",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
                )
                applied.append(letter)
                p_idx += 1

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    # Convert to strict 8-bit RGB (no alpha channel) + 600 DPI + LZW compression
    labeled_im = Image.open(buf).convert("RGB")
    joms_name = f"Fig{idx}.tif" if idx <= 11 else f"FigS{idx - 11}.tif"

    labeled_im.save(os.path.join("figures", fname), dpi=(600, 600), compression="tiff_lzw")
    labeled_im.save(os.path.join("joms_submission_figures", joms_name), dpi=(600, 600), compression="tiff_lzw")
    all_frames.append(labeled_im)

    # Save individual PNG preview (opens natively in VS Code & Windows Photos)
    pw = 1100
    ph = int(h * (pw / w))
    prev = labeled_im.resize((pw, ph), Image.Resampling.LANCZOS)
    prev.save(os.path.join("preview_pngs", f"{idx:02d}_{joms_name.replace('.tif', '.png')}"))

    # Embed preview in self-contained HTML gallery
    jbuf = io.BytesIO()
    prev.save(jbuf, format="JPEG", quality=88)
    b64 = base64.b64encode(jbuf.getvalue()).decode("ascii")
    label_str = ", ".join(applied) if applied else "Single panel (no subpanel letter needed)"
    html_cards.append(
        f"<div style='background:#fff;padding:16px;margin-bottom:24px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.15)'>"
        f"<h3 style='margin:0 0 8px;font-family:Arial'>{joms_name} &mdash; {fname} <span style='color:#0b66c3'>[{n_rows}x{n_cols}: {label_str}]</span></h3>"
        f"<img src='data:image/jpeg;base64,{b64}' style='max-width:100%;border:1px solid #ddd'/></div>"
    )
    print(f"{idx:02d}.  {fname:<52} {n_rows}x{n_cols:<5} {label_str}")

# Rebuild 14-page 600-DPI LZW TIFF bundle
all_frames[0].save(
    "JOMS_All_14_Figures_Multipage_600DPI.tiff",
    save_all=True,
    append_images=all_frames[1:],
    dpi=(600, 600),
    compression="tiff_lzw",
)

with open("view_all_14_figures.html", "w") as f:
    f.write(
        "<!DOCTYPE html><html><body style='background:#f2f4f7;padding:24px;font-family:Arial'>"
        "<h2>JoMS Labeled Figures Verification (600 DPI RGB LZW TIFFs Ready)</h2>"
        + "\n".join(html_cards)
        + "</body></html>"
    )

print("\nSuccess! Download 'view_all_14_figures.html' to view all 14 figures in your browser.")
