"""
Interactive crop region selector for RealSense camera.
Click and drag to define the detection region.

Controls:
  - Left click + drag: Draw crop rectangle
  - R: Reset rectangle
  - S: Save and print crop coordinates
  - Q/ESC: Quit
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


@dataclass
class Config:
    WIDTH: int = 1280
    HEIGHT: int = 720
    FPS: int = 30
    OUTPUT_FILE: Path = Path("/home/jeong/perception/app/data/crop_config.json")


CFG = Config()

# Global state for mouse callback
drawing = False
start_x, start_y = 0, 0
end_x, end_y = 0, 0
rect_defined = False


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y, rect_defined
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        end_x, end_y = x, y
        rect_defined = False
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_x, end_y = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        rect_defined = True


def get_normalized_rect() -> tuple[int, int, int, int]:
    """Return rectangle with (x1, y1, x2, y2) where x1 < x2 and y1 < y2."""
    x1 = min(start_x, end_x)
    y1 = min(start_y, end_y)
    x2 = max(start_x, end_x)
    y2 = max(start_y, end_y)
    return x1, y1, x2, y2


def main():
    global start_x, start_y, end_x, end_y, rect_defined, drawing
    
    print("=" * 60)
    print("Crop Region Selector")
    print("=" * 60)
    print("Controls:")
    print("  Left click + drag: Draw crop rectangle")
    print("  R: Reset rectangle")
    print("  S: Save crop coordinates")
    print("  Q/ESC: Quit")
    print("=" * 60)
    
    # Initialize RealSense
    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, CFG.WIDTH, CFG.HEIGHT, rs.format.bgr8, CFG.FPS)
    pipeline.start(rs_cfg)
    
    # Warmup
    for _ in range(30):
        pipeline.wait_for_frames()
    
    # Create window
    cv2.namedWindow("Crop Selector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Crop Selector", mouse_callback)
    
    saved_crop = None
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            img = np.asanyarray(color_frame.get_data())
            display = img.copy()
            
            # Draw current rectangle
            if drawing or rect_defined:
                x1, y1, x2, y2 = get_normalized_rect()
                
                # Draw semi-transparent overlay outside the rectangle
                overlay = display.copy()
                overlay[:y1, :] = (overlay[:y1, :] * 0.4).astype(np.uint8)  # Top
                overlay[y2:, :] = (overlay[y2:, :] * 0.4).astype(np.uint8)  # Bottom
                overlay[y1:y2, :x1] = (overlay[y1:y2, :x1] * 0.4).astype(np.uint8)  # Left
                overlay[y1:y2, x2:] = (overlay[y1:y2, x2:] * 0.4).astype(np.uint8)  # Right
                display = overlay
                
                # Draw rectangle border
                color = (0, 255, 0) if rect_defined else (0, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                
                # Show dimensions
                w, h = x2 - x1, y2 - y1
                text = f"({x1}, {y1}) -> ({x2}, {y2})  |  {w}x{h}"
                cv2.putText(display, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Instructions
            cv2.putText(display, "Drag to select region | R=Reset | S=Save | Q=Quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if saved_crop:
                cv2.putText(display, f"Saved: {saved_crop}", 
                           (10, CFG.HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Crop Selector", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            
            elif key == ord('r'):  # Reset
                start_x, start_y = 0, 0
                end_x, end_y = 0, 0
                rect_defined = False
                drawing = False
                saved_crop = None
                print("[Reset] Rectangle cleared")
            
            elif key == ord('s'):  # Save
                if rect_defined:
                    x1, y1, x2, y2 = get_normalized_rect()
                    crop_config = {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "source_width": CFG.WIDTH,
                        "source_height": CFG.HEIGHT,
                    }
                    
                    CFG.OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
                    with open(CFG.OUTPUT_FILE, "w") as f:
                        json.dump(crop_config, f, indent=2)
                    
                    saved_crop = f"({x1}, {y1}, {x2}, {y2})"
                    print(f"\n[Saved] Crop config to: {CFG.OUTPUT_FILE}")
                    print(f"  x1, y1 = {x1}, {y1}")
                    print(f"  x2, y2 = {x2}, {y2}")
                    print(f"  size   = {x2-x1} x {y2-y1}")
                    print(f"\nUse in 3.py:")
                    print(f"  CROP_X1, CROP_Y1 = {x1}, {y1}")
                    print(f"  CROP_X2, CROP_Y2 = {x2}, {y2}")
                else:
                    print("[Warning] Draw a rectangle first!")
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    
    print("\n[Done]")


if __name__ == "__main__":
    main()
