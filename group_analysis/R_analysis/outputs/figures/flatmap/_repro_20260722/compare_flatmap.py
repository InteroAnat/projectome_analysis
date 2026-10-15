import hashlib, re, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ORIG = Path(r"D:/projectome_analysis/group_analysis/R_analysis/outputs/figures/flatmap")
REPRO = ORIG / "_repro_20260722"
FILES = [
    "flatmap_all_monkeys_combined.png",
    "flatmap_all_monkeys_combined.svg",
    "flatmap_per_monkey_panels.png",
    "flatmap_per_monkey_panels.svg",
]

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

from PIL import Image

def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def compare_png(a, b):
    ia = np.array(Image.open(a).convert("RGBA"))
    ib_img = Image.open(b).convert("RGBA")
    ib = np.array(ib_img)
    shape_note = "same dims"
    if ia.shape != ib.shape:
        ib = np.array(ib_img.resize((ia.shape[1], ia.shape[0]), Image.Resampling.LANCZOS))
        shape_note = f"repro resized {ib_img.size} -> {ia.shape[1], ia.shape[0]}"
    diff = ia.astype(np.int16) - ib.astype(np.int16)
    mse = float(np.mean(diff ** 2))
    pct_diff = 100.0 * float(np.any(diff != 0, axis=-1).mean())
    ssim_val = None
    if HAS_SSIM and ia.shape == ib.shape:
        ssim_val = float(ssim_fn(ia, ib, channel_axis=-1, data_range=255))
    identical = np.array_equal(ia, ib)
    return {
        "orig_wh": Image.open(a).size,
        "repro_wh": Image.open(b).size,
        "shape_note": shape_note,
        "mse": mse,
        "pct_diff_pixels": pct_diff,
        "ssim": ssim_val,
        "byte_identical": identical,
    }

def svg_stats(p):
    text = p.read_text(encoding="utf-8", errors="replace")
    norm = re.sub(r'id="[^"]*"', 'id="X"', text)
    norm = re.sub(r'\s+', ' ', norm).strip()
    try:
        root = ET.fromstring(text)
        tag_counts = {}
        for el in root.iter():
            tag_counts[el.tag] = tag_counts.get(el.tag, 0) + 1
    except ET.ParseError as e:
        tag_counts = {"parse_error": str(e)}
    return {
        "bytes": p.stat().st_size,
        "sha256": file_sha256(p),
        "norm_sha256": hashlib.sha256(norm.encode()).hexdigest()[:16],
        "tag_counts": tag_counts,
        "line_count": text.count("\n") + 1,
    }

def verdict_png(c, sha_match):
    if sha_match:
        return "identical"
    if c.get("ssim") is not None and c["ssim"] >= 0.995 and c["pct_diff_pixels"] < 1.0:
        return "near-identical"
    if c.get("ssim") is not None and c["ssim"] >= 0.98:
        return "near-identical"
    if c["pct_diff_pixels"] < 5 and c["mse"] < 500:
        return "near-identical"
    return "divergent"

def verdict_svg(o, r):
    if o["sha256"] == r["sha256"]:
        return "identical"
    if o["norm_sha256"] == r["norm_sha256"]:
        return "near-identical (norm)"
    if o["tag_counts"] == r["tag_counts"] and abs(o["bytes"] - r["bytes"]) / max(o["bytes"], 1) < 0.05:
        return "near-identical (structure)"
    return "divergent"

print("HAS_SSIM", HAS_SSIM)
print("| file | orig_bytes | repro_bytes | dims | MSE | SSIM | pct_diff_px | sha_match | verdict |")
print("|------|------------|-------------|------|-----|------|-------------|-----------|---------|")

all_pass = True
any_partial = False
for fn in FILES:
    o = ORIG / fn
    r = REPRO / fn
    ob, rb = o.stat().st_size, r.stat().st_size
    sha_o, sha_r = file_sha256(o), file_sha256(r)
    sha_match = sha_o == sha_r
    if fn.endswith(".png"):
        c = compare_png(o, r)
        v = verdict_png(c, sha_match)
        ssim_s = f"{c['ssim']:.6f}" if c["ssim"] is not None else "n/a"
        dim = f"{c['orig_wh']} vs {c['repro_wh']} ({c['shape_note']})"
        print(f"| {fn} | {ob} | {rb} | {dim} | {c['mse']:.2f} | {ssim_s} | {c['pct_diff_pixels']:.3f}% | {sha_match} | {v} |")
        if v == "divergent":
            all_pass = False
        elif v != "identical":
            any_partial = True
    else:
        so, sr = svg_stats(o), svg_stats(r)
        v = verdict_svg(so, sr)
        print(f"| {fn} | {ob} | {rb} | SVG lines {so['line_count']} vs {sr['line_count']} | n/a | n/a | n/a | {sha_match} | {v} |")
        print(f"  orig_sha={so['sha256']} repro_sha={sr['sha256']} norm_match={so['norm_sha256']==sr['norm_sha256']}")
        if "divergent" in v:
            all_pass = False
        elif not v.startswith("identical"):
            any_partial = True

if all_pass and not any_partial:
    overall = "PASS"
elif all_pass and any_partial:
    overall = "PARTIAL"
else:
    overall = "FAIL"
print("OVERALL", overall)
print("ORIG_DIR", ORIG)
print("REPRO_DIR", REPRO)
