"""Copy-paste synthesis of bubble-wrap blocks for detector training.

1. LaMa (IOPaint CLI) removes every segmented block -> clean background plate.
2. Each SAM2 mask becomes an RGBA cutout.
3. For every synthetic image, 2-7 cutouts are pasted at random position / rotation /
   scale (OBJ_SCALE) / brightness inside the table region, without overlapping, then the
   whole image is zoomed (ZOOM) to mimic a different camera height. Labels come
   for free from the pasted alpha: annotations.json in the ai/annotate.py format
   {file: [[x1, y1, x2, y2], ...]}.

usage: python synth_blocks.py <masks.npz> <out_dir> [n_images]
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

IOPAINT = str(Path.home() / ".local/bin/iopaint")

# Where blocks may be placed in this top-camera view (x0, y0, x1, y1), and areas to keep clear.
TABLE = (440, 100, 1180, 680)
OBJ_SCALE = (0.7, 1.4)      # per-block size variation
ZOOM = (0.75, 1.6)          # whole-image zoom: camera closer (>1) or farther (<1)
MIN_VISIBLE = 0.25          # keep partly cropped blocks labelled; an unlabelled visible block teaches "not a block"

KEEP_OUT = [(1070, 370, 1170, 580),   # power strip
            (740, 0, 900, 160)]       # gripper


def make_plate(img, masks, work: Path):
    """Remove all blocks with LaMa; returns the inpainted background."""
    for d in ("in", "mask", "out"):
        (work / d).mkdir(parents=True, exist_ok=True)
    union = np.any(masks, axis=0).astype(np.uint8) * 255
    # LaMa leaves dark ghost blobs when the hole hugs the object; ~50 px of margin fills cleanly
    union = cv2.dilate(union, np.ones((51, 51), np.uint8))
    cv2.imwrite(str(work / "in/plate.png"), img)
    cv2.imwrite(str(work / "mask/plate.png"), union)
    subprocess.run([IOPAINT, "run", "--model=lama", "--device=cuda", f"--image={work / 'in'}",
                    f"--mask={work / 'mask'}", f"--output={work / 'out'}"],
                   check=True, capture_output=True)
    return cv2.imread(str(work / "out/plate.png")), union


def cutouts(img, masks, pad=6):
    """RGBA crops of each block; alpha feathered inward so no table halo is carried along."""
    outs = []
    for m in masks:
        ys, xs = np.nonzero(m)
        y0, y1 = max(ys.min() - pad, 0), ys.max() + pad + 1
        x0, x1 = max(xs.min() - pad, 0), xs.max() + pad + 1
        a = cv2.erode(m[y0:y1, x0:x1].astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
        a = cv2.GaussianBlur(a, (5, 5), 0)
        outs.append(np.dstack([img[y0:y1, x0:x1], a]))
    return outs


def transform(rgba, angle, scale, gain):
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c) + 2, int(h * c + w * s) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    out[..., :3] = np.clip(out[..., :3].astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return out


def overlaps(b, others, gap=6):
    return any(not (b[2] + gap < o[0] or o[2] + gap < b[0] or b[3] + gap < o[1] or o[3] + gap < b[1])
               for o in others)


def synthesize(plate, cuts, rng, n_min=2, n_max=7):
    out = plate.astype(np.float32)
    boxes = []
    for _ in range(rng.integers(n_min, n_max + 1)):
        piece = transform(cuts[rng.integers(len(cuts))], rng.uniform(0, 360),
                          rng.uniform(*OBJ_SCALE), rng.uniform(0.92, 1.08))
        h, w = piece.shape[:2]
        for _try in range(50):
            x = int(rng.integers(TABLE[0], TABLE[2] - w))
            y = int(rng.integers(TABLE[1], TABLE[3] - h))
            alpha = piece[..., 3].astype(np.float32) / 255
            ys, xs = np.nonzero(alpha > 0.5)
            b = (x + xs.min(), y + ys.min(), x + xs.max(), y + ys.max())
            if not overlaps(b, boxes + KEEP_OUT):
                break
        else:
            continue
        # faint contact shadow, offset away from the light like in the real frame
        sh = cv2.GaussianBlur(alpha, (9, 9), 0) * 0.18
        sy, sx = y + 3, x + 3
        out[sy:sy + h, sx:sx + w] *= (1 - sh[..., None])
        roi = out[y:y + h, x:x + w]
        roi[:] = roi * (1 - alpha[..., None]) + piece[..., :3].astype(np.float32) * alpha[..., None]
        boxes.append(tuple(int(v) for v in b))
    return out.astype(np.uint8), [list(b) for b in boxes]


def zoom(img, boxes, z, rng):
    """Whole-image zoom that keeps the output size. z > 1 crops a random window of the
    enlarged image; z < 1 shrinks it onto a canvas of the scene's mean colour (mirror
    padding would duplicate blocks without labels). Boxes follow the same transform."""
    H, W = img.shape[:2]
    big = cv2.resize(img, (round(W * z), round(H * z)), interpolation=cv2.INTER_LINEAR)
    bh, bw = big.shape[:2]
    if z >= 1:
        ox, oy = int(rng.integers(0, bw - W + 1)), int(rng.integers(0, bh - H + 1))
        out = big[oy:oy + H, ox:ox + W]
        dx, dy = -ox, -oy
    else:
        out = np.empty_like(img)
        out[:] = img.reshape(-1, 3).mean(0).astype(np.uint8)
        dx, dy = int(rng.integers(0, W - bw + 1)), int(rng.integers(0, H - bh + 1))
        out[dy:dy + bh, dx:dx + bw] = big
    kept = []
    for x1, y1, x2, y2 in boxes:
        b = [x1 * z + dx, y1 * z + dy, x2 * z + dx, y2 * z + dy]
        c = [max(b[0], 0), max(b[1], 0), min(b[2], W - 1), min(b[3], H - 1)]
        if c[2] <= c[0] or c[3] <= c[1]:
            continue
        if (c[2] - c[0]) * (c[3] - c[1]) >= MIN_VISIBLE * (b[2] - b[0]) * (b[3] - b[1]):
            kept.append([int(round(v)) for v in c])
    return out, kept


def main():
    npz_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    n_images = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    data = np.load(npz_path, allow_pickle=True)
    img = cv2.imread(str(data["image"]))
    masks = data["masks"]
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    plate, union = make_plate(img, masks, out_dir / "_lama")
    cv2.imwrite(str(out_dir / "plate.png"), plate)
    cuts = cutouts(img, masks)

    rng = np.random.default_rng(0)
    ann, previews = {}, []
    for i in range(n_images):
        syn, boxes = synthesize(plate, cuts, rng)
        z = float(rng.uniform(*ZOOM))
        syn, boxes = zoom(syn, boxes, z, rng)
        name = f"synth_{i:03d}.png"
        cv2.imwrite(str(out_dir / "images" / name), syn)
        ann[name] = boxes
        pv = syn.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(pv, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(pv, f"{name}  zoom x{z:.2f}  {len(boxes)} labels", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        previews.append(cv2.resize(pv, (640, 360)))
    (out_dir / "annotations.json").write_text(json.dumps(ann, indent=1))

    # preview: original | LaMa plate, then the synthetic images with their auto labels
    top = [cv2.resize(img, (640, 360)), cv2.resize(plate, (640, 360))]
    for t, label in zip(top, ("original (3 real blocks)", "LaMa: blocks removed")):
        cv2.putText(t, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    tiles = top + previews
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)])
    cv2.imwrite(str(out_dir / "preview.png"), grid)
    print(f"{n_images} images, {sum(len(v) for v in ann.values())} labels -> {out_dir}")


if __name__ == "__main__":
    main()
