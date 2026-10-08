"""The basic-task feedback loop (diagrams/basic_task_flow.png):

 1 scan_box -> 2 split_partition -> [ 3 pick -> 4 holding? -> 5 move above cell -> 6 check
 -> 7 place + verify ] repeated per block; failures jump back as in the diagram.
Every step reports to the run log (and dashboard): step tracker, metrics, live overlay.
"""
import numpy as np

from . import perception as P


class StopRequested(Exception):
    pass


class Pipeline:
    def __init__(self, cfg, cams, hub, robot, calib, log):
        self.cfg, self.cams, self.hub, self.robot, self.calib, self.log = cfg, cams, hub, robot, calib, log
        self.block_h = cfg["motion"]["block_height_m"]
        self.table_z = calib.table_z
        self.picked = []          # offline (static image) only: blocks already used, so they are skipped
        self.retries = 0

    def _check_stop(self):
        if self.log.stop_requested:
            raise StopRequested()

    def _step(self, n, status, note=""):
        self.log.step(n, status, note)
        if status == "running":
            self.log.overlay(step_text=f"STEP {n} · {note}")
        self._check_stop()

    # ------------------------------------------------------------------ steps
    def step1_scan_box(self):
        self._step(1, "running", "Grounding DINO + VGGT + RealSense")
        f = self.cams.top()
        box = P.scan_box(f, self.hub, self.cfg, self.log)
        msg = f"box state = {box.state}"
        if box.height_m:
            msg += f", rim {box.height_m*100:.1f} cm (VGGT {box.height_vggt_m*100:.1f} cm)"
        self.log.info("1. " + msg)
        self._step(1, "ok" if box.state == "partition_in_box" else "fail", msg)
        return f, box

    def step2_partition(self, f, box):
        self._step(2, "running", "Canny + RANSAC + DINOv3")
        part = P.split_partition(f, box, self.hub, self.cfg, self.log)
        states = {}
        for c in part.cells:
            states[c.state] = states.get(c.state, 0) + 1
        self.log.info(f"2. partition {part.rows}x{part.cols}, cells: {states}")
        self._step(2, "ok", f"{part.rows}×{part.cols}, 빈 칸 {states.get('empty', 0)}")
        return part

    def z_safe(self, box):
        return self.table_z + box.height_m + self.block_h + self.cfg["motion"]["safe_margin_m"]

    def step3_pick(self, g, box):
        self._step(3, "running", f"블록 ({g.center_px[0]:.0f},{g.center_px[1]:.0f}) 점수 {g.score:.2f}")
        m = self.cfg["motion"]
        x, y = self.calib.to_robot(*g.center_px)
        yaw = self.calib.yaw_to_robot(*g.center_px, g.jaw_deg)
        z_grasp = self.table_z + self.block_h - m["grasp_depth_m"]
        r = self.robot
        r.open()
        r.move_to(x, y, self.z_safe(box), yaw)
        r.move_to(x, y, z_grasp, yaw)
        if hasattr(r, "over_block"):                                  # mock: pretend a block is there
            r.over_block = max(15.0, g.opening_px * 0.6)
        r.close()
        r.move_to(x, y, z_grasp + m["lift_m"], yaw)
        self._step(3, "ok", f"x={x:.3f} y={y:.3f} m yaw={np.degrees(yaw):.0f}°")
        return self.step4_holding()

    def step4_holding(self):
        self._step(4, "running", "그리퍼 폭")
        w = self.robot.gripper_width_mm()
        ok = w > self.cfg["robot"]["holding_min_mm"]
        # TODO: + wrist D405 check (depth pixels between the fingers) once mounted
        self.log.metric(gripper_mm=round(w, 1))
        self.log.info(f"4. holding? gripper width {w:.0f} mm -> {'yes' if ok else 'NO'}")
        self._step(4, "ok" if ok else "fail", f"그리퍼 {w:.0f} mm")
        if not ok:
            self.retries += 1
            self.log.metric(retries=self.retries)
        return ok

    def step5_move_above(self, cell, box):
        self._step(5, "running", f"칸 {cell.id}")
        x, y = self.calib.to_robot(*cell.center_px)
        yaw = self.calib.yaw_to_robot(*cell.center_px, cell.yaw_deg)
        self.robot.move_to(x, y, self.z_safe(box), yaw)
        self._step(5, "ok", f"칸 {cell.id} 위 z={self.z_safe(box):.3f} m")
        return x, y, yaw

    def step6_check(self):
        self._step(6, "running", "파지 유지 확인")
        # TODO: top D435i gripper position vs target cell, wrist D405 alignment (visual servoing)
        w = self.robot.gripper_width_mm()
        ok = w > self.cfg["robot"]["holding_min_mm"]
        self.log.metric(gripper_mm=round(w, 1))
        self.log.info(f"6. above target: gripper {w:.0f} mm -> holding={ok} (position check: TODO)")
        self._step(6, "ok" if ok else "fail", f"그리퍼 {w:.0f} mm, 위치 확인 TODO")
        return ok

    def step7_place_and_verify(self, cell, xy_yaw, box):
        self._step(7, "running", f"칸 {cell.id}에 내려놓기")
        m = self.cfg["motion"]
        x, y, yaw = xy_yaw
        floor = self.table_z + 0.005                                  # cell floor ~ table + cardboard
        z_release = floor + m["release_above_floor_m"] + self.block_h - m["grasp_depth_m"]
        self.robot.move_to(x, y, z_release, yaw)
        self.robot.open()
        if hasattr(self.robot, "over_block"):
            self.robot.over_block = None
        self.robot.move_to(x, y, self.z_safe(box), yaw)
        if self.cams.static:
            state = "seated"
            self.log.info(f"7. cell {cell.id}: verification simulated as seated (offline image)")
        else:
            f = self.cams.top(with_depth=False)
            state, conf = P.cell_state(f, cell, self.hub, self.cfg, self.log)
            self.log.info(f"7. cell {cell.id}: {state} ({conf:.2f})")
        self._step(7, "ok" if state == "seated" else "fail", f"칸 {cell.id}: {state}")
        return state

    # ------------------------------------------------------------------ loop
    def run(self, max_blocks=None):
        try:
            return self._run(max_blocks)
        except StopRequested:
            self.log.info("STOPPED from the dashboard", "error")
            self.robot.stop()
            return {"placed": self.log_placed, "stopped": "dashboard stop"}

    def _run(self, max_blocks=None):
        cfg = self.cfg
        max_blocks = max_blocks or cfg["run"]["max_blocks"]
        self.log_placed = 0
        self.robot.connect()
        f, box = self.step1_scan_box()
        if box.state != "partition_in_box":
            self.log.info("STOP: the partition is not in a box", "warn")
            return {"placed": 0, "stopped": box.state}
        part = self.step2_partition(f, box)
        self.log.json("partition", [c.__dict__ for c in part.cells])

        placed, attempts = 0, 0
        while placed < max_blocks:
            empty = part.empty_cells()
            if not empty:
                self.log.info("no empty cell left")
                break
            self._step(3, "running", "블록 찾기")
            blocks = P.find_blocks(self.cams.top(with_depth=False), box, self.hub, cfg, self.log)
            if self.cams.static:
                blocks = [b for b in blocks if all(np.hypot(b.center_px[0] - u, b.center_px[1] - v) > 15
                                                   for u, v in self.picked)]
            if not blocks:
                self.log.info("no block left in the supply area")
                break
            target = empty[0]
            g = blocks[0]
            self.log.metric(target_cell=target.id)
            self.log.overlay(target_cell=target.id)
            self.log.info(f"--- block {placed + 1}: grasp at px ({g.center_px[0]:.0f},{g.center_px[1]:.0f}) "
                          f"score {g.score:.2f} -> cell {target.id}")

            if not any(self.step3_pick(g, box) for _ in range(cfg["retry"]["pick"])):
                self.log.warn("pick failed after retries -> look for blocks again")
                attempts += 1
                if attempts > max_blocks + cfg["retry"]["pick"]:
                    break
                if self.cams.static:
                    self.picked.append(g.center_px)      # offline: give up on this block
                continue

            state = "dropped"
            for _ in range(cfg["retry"]["place"]):
                xy_yaw = self.step5_move_above(target, box)
                if not self.step6_check():
                    state = "dropped"
                    break
                state = self.step7_place_and_verify(target, xy_yaw, box)
                if state != "straddle":
                    break
                self.log.warn("straddle -> re-grasp rotated 90 deg and re-insert")
                g = type(g)(target.center_px, g.jaw_deg + 90, g.opening_px, g.length_px, g.score, g.box_xyxy)
                self.retries += 1
                self.log.metric(retries=self.retries)
                self.step3_pick(g, box)

            if state == "seated":
                target.state = "seated"
                placed += 1
                self.log_placed = placed
                self.picked.append(g.center_px)
                self.log.metric(placed=placed)
                self.log.overlay(cells=P.cells_overlay(part.cells))
            elif state in ("dropped", "empty"):
                self.log.warn(f"block lost ({state}) -> back to step 3")
            else:
                target.state = state                                   # leave uncertain cells alone
                self.log.overlay(cells=P.cells_overlay(part.cells))
        self.robot.home()
        self.log.info(f"DONE: {placed} block(s) seated")
        return {"placed": placed, "grid": f"{part.rows}x{part.cols}", "cells": {c.id: c.state for c in part.cells}}
