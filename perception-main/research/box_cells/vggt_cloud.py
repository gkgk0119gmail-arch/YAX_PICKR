"""Single-image VGGT -> depth map + point cloud for the top-down RGB frame.

VGGT's output scale is arbitrary, so the depth is rescaled to metres by matching
the median table distance to the RealSense depth (depth.raw) over the image centre.
"""
import sys

import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

img_path = sys.argv[1] if len(sys.argv) > 1 else "cam4.jpg"
device = "cuda"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B-Commercial").to(device).eval()
images = load_and_preprocess_images([img_path]).to(device)          # 1x3xHxW, W=518
print("input", tuple(images.shape))

with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
    pred = model(images[None])
extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
depth = pred["depth"][0, 0, ..., 0].float().cpu().numpy()            # HxW, VGGT units
conf = pred["depth_conf"][0, 0].float().cpu().numpy()
points = unproject_depth_map_to_point_map(pred["depth"][0].float().cpu().numpy(),
                                          extri[0].float().cpu().numpy(),
                                          intri[0].float().cpu().numpy())[0]   # HxWx3
K = intri[0, 0].float().cpu().numpy()

# metric scale from the RealSense depth (median over the central region, mostly table + box)
rs = np.fromfile((sys.argv[2] if len(sys.argv) > 2 else "depth.raw"), np.uint16).reshape(-1, 480, 848)[-1].astype(np.float32) / 1000.0
rs_c = rs[140:340, 250:600]
h, w = depth.shape
v_c = depth[int(0.29 * h):int(0.71 * h), int(0.29 * w):int(0.71 * w)]
scale = np.median(rs_c[rs_c > 0]) / np.median(v_c)
print(f"vggt depth range {depth.min():.3f}-{depth.max():.3f}, scale to metres x{scale:.3f}")

np.savez(sys.argv[3] if len(sys.argv) > 3 else "vggt_out.npz", depth_m=depth * scale, conf=conf, points_m=points * scale, K=K,
         image=images[0].permute(1, 2, 0).cpu().numpy())

# coloured point cloud for viewing (PLY, high-confidence points only)
rgb = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
P = (points * scale).reshape(-1, 3)
keep = conf.reshape(-1) > np.percentile(conf, 10)
with open(sys.argv[4] if len(sys.argv) > 4 else "vggt_cloud.ply", "w") as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {keep.sum()}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for (x, y, z), (r, g, b) in zip(P[keep], rgb[keep]):
        f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
print("saved vggt_out.npz, vggt_cloud.ply", keep.sum(), "points")
