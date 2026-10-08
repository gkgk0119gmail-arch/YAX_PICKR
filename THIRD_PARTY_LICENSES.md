# Third-party models and libraries

Model weights are **not** stored in this repository. They are downloaded from the original source
(`setup/get_models.py`). Summary as of 2026-10 (read the original license before any redistribution;
this is not legal advice).

| Component | Used for | License | Commercial use | Notes |
|---|---|---|---|---|
| Grounding DINO base (`IDEA-Research/grounding-dino-base`) | box / partition / gripper detection, auto labels | Apache-2.0 | ✅ | keep license + notice when redistributing |
| SAM 2.1 (`facebook/sam2.1-hiera-large`) | segmentation | Apache-2.0 | ✅ | |
| RF-DETR (Roboflow, `rfdetr`) | bubble-wrap block detector (fine-tuned by us) | Apache-2.0 | ✅ | our fine-tuned weights: GitHub Release `models-v1` |
| DINOv3 ViT-B/16 (`facebook/dinov3-vitb16-pretrain-lvd1689m`) | cell-state features | DINOv3 License | ✅ with conditions | gated download; no military/weapons/ITAR use; ship a copy of the license when redistributing |
| VGGT-1B-Commercial (`facebook/VGGT-1B-Commercial`) | depth / point map from one RGB image | VGGT License + AUP | ✅ with conditions | gated download. **Do not use `facebook/VGGT-1B` (CC BY-NC 4.0, non-commercial)** |
| LaMa via IOPaint | inpainting for synthetic training data | Apache-2.0 / MIT | ✅ | data preparation only |
| piper_sdk (AgileX) | PiPER control over CAN | see AgileX repository | - | |
| transformers, PyTorch, OpenCV, scikit-learn, SciPy | runtime | Apache-2.0 / BSD | ✅ | |
| Qwen3-VL 2B/4B/8B (optional) | VLM checks, voice commands (planned) | Apache-2.0 | ✅ | Qwen2.5-VL-3B is non-commercial — not used |
