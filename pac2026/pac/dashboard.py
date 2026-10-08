"""Local monitoring dashboard (stdlib only): http://<this-pc>:8800

  /              page (polls /state every 0.5 s)
  /state         json: run status, step tracker, metrics, panel versions, recent log, run history
  /panel/<k>.jpg latest image of a model card (scene, height, lines, cells, blocks, verify)
  /live.mjpg     live camera with the current recognition overlaid
  POST /stop     sets stop_flag (the pipeline stops the robot at the next check)
"""
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

STEPS = {1: "상자 위치·높이", 2: "파티션 n×m·빈 칸", 3: "블록 찾기·집기", 4: "파지 확인",
         5: "목표 칸 위 이동", 6: "위치·파지 재확인", 7: "내려놓기·안착 판정"}
PANELS = [("scene", "Grounding DINO — 상자 / 파티션"), ("height", "VGGT 높이 지도"),
          ("lines", "Canny + RANSAC → 칸 분할"), ("cells", "DINOv3 칸 판정"),
          ("blocks", "RF-DETR + SAM2 + RANSAC — 블록·파지"), ("verify", "넣은 뒤 판정")]
CELL_COLORS = {"empty": (200, 200, 200), "seated": (0, 200, 0), "straddle": (0, 0, 255),
               "uncertain": (0, 200, 255), "out_of_view": (255, 0, 255)}


class Dashboard:
    def __init__(self, runs_dir: Path, port=8800):
        self.runs_dir, self.port = Path(runs_dir), port
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.frame_source = None                     # callable -> BGR frame for /live.mjpg
        self.state = {"run_id": None, "status": "idle", "started": None, "steps": {}, "metrics": {},
                      "panels": {}, "info": {}}
        self.logs = deque(maxlen=400)
        self.images = {}
        self.overlay = {}

    # ------------------------------------------------------------------ updates from the pipeline
    def start_run(self, run_id):
        with self.lock:
            self.state.update(run_id=run_id, status="running", started=time.time(), metrics={},
                              steps={n: {"status": "pending", "note": "", "ok": 0, "fail": 0} for n in STEPS})
            self.logs.clear()
            self.overlay = {}
        self.stop_flag.clear()

    def set_info(self, **kv):
        with self.lock:
            self.state["info"].update(kv)

    def set_status(self, status):
        with self.lock:
            self.state["status"] = status
            if status != "running":                  # nothing is running any more
                for s in self.state["steps"].values():
                    if s["status"] == "running":
                        s["status"] = "done"

    def log(self, t, msg, level="info"):
        with self.lock:
            self.logs.append({"t": round(t, 2), "msg": msg, "level": level})

    def set_step(self, n, status, note=""):
        with self.lock:
            s = self.state["steps"].setdefault(n, {"status": "pending", "note": "", "ok": 0, "fail": 0})
            for k in list(self.state["steps"]):
                if k != n and self.state["steps"][k]["status"] == "running":
                    self.state["steps"][k]["status"] = "done"
            s["status"], s["note"] = status, note
            if status == "ok":
                s["ok"] += 1
            elif status == "fail":
                s["fail"] += 1

    def set_metrics(self, **kv):
        with self.lock:
            m = self.state["metrics"]
            for k, v in kv.items():
                if isinstance(v, dict) and isinstance(m.get(k), dict):
                    m[k].update(v)
                else:
                    m[k] = v

    def set_panel(self, key, img, caption=None):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with self.lock:
            self.images[key] = buf.tobytes()
            p = self.state["panels"].setdefault(key, {"v": 0})
            p["v"] += 1
            p["caption"] = caption or ""

    def set_overlay(self, **kv):
        with self.lock:
            self.overlay.update(kv)

    # ------------------------------------------------------------------ live view
    def draw_overlay(self, img):
        with self.lock:
            ov = dict(self.overlay)
        out = img.copy()
        if ov.get("box"):
            x1, y1, x2, y2 = map(int, ov["box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for c in ov.get("cells", []):
            col = CELL_COLORS.get(c["state"], (255, 255, 255))
            q = np.round(np.asarray(c["quad"])).astype(np.int32)
            thick = 4 if c["id"] == ov.get("target_cell") else 2
            cv2.polylines(out, [q], True, col, thick)
            x, y = q.mean(0).astype(int)
            cv2.putText(out, str(c["id"]), (x - 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        for i, g in enumerate(ov.get("grasps", [])):
            col = (255, 0, 255) if i == ov.get("grasp_index", 0) else (200, 120, 200)
            x1, y1, x2, y2 = map(int, g["box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cx, cy = g["center"]
            a = np.radians(g["jaw"])
            h = g["opening"] / 2 + 8
            cv2.line(out, (int(cx - h * np.cos(a)), int(cy - h * np.sin(a))),
                     (int(cx + h * np.cos(a)), int(cy + h * np.sin(a))), (0, 0, 255), 3)
        step = ov.get("step_text")
        if step:
            cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(out, step, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return out

    def live_frames(self):
        while True:
            img = self.frame_source() if self.frame_source else None
            if img is None:
                img = np.full((360, 640, 3), 40, np.uint8)
                cv2.putText(img, "no camera", (230, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            img = self.draw_overlay(img)
            img = cv2.resize(img, (960, int(960 * img.shape[0] / img.shape[1])))
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield buf.tobytes()
            time.sleep(0.1)

    # ------------------------------------------------------------------ state for the page
    def history(self):
        rows = []
        for d in sorted(self.runs_dir.glob("*/"), reverse=True)[:12]:
            r = d / "result.json"
            res = json.loads(r.read_text()) if r.exists() else {}
            rows.append({"run": d.name, "placed": res.get("placed"), "stopped": res.get("stopped"),
                         "grid": res.get("grid")})
        return rows

    def snapshot(self):
        with self.lock:
            s = json.loads(json.dumps(self.state, default=str))
            s["logs"] = list(self.logs)[-150:]
        s["elapsed"] = round(time.time() - s["started"], 1) if s.get("started") and s["status"] == "running" else None
        s["history"] = self.history()
        s["stop_requested"] = self.stop_flag.is_set()
        s["step_names"] = STEPS
        s["panel_titles"] = dict(PANELS)
        return s

    # ------------------------------------------------------------------ server
    def serve(self):
        dash = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif self.path.startswith("/state"):
                    self._send(200, json.dumps(dash.snapshot(), ensure_ascii=False).encode(), "application/json")
                elif self.path.startswith("/panel/"):
                    key = self.path.split("/")[2].split(".")[0]
                    with dash.lock:
                        b = dash.images.get(key)
                    self._send(200, b, "image/jpeg") if b else self._send(404, b"", "text/plain")
                elif self.path.startswith("/live.mjpg"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        for jpg in dash.live_frames():
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self._send(404, b"", "text/plain")

            def do_POST(self):
                if self.path == "/stop":
                    dash.stop_flag.set()
                    dash.log(0, "STOP requested from the dashboard", "error")
                    self._send(200, b"ok", "text/plain")
                else:
                    self._send(404, b"", "text/plain")

        srv = ThreadingHTTPServer(("0.0.0.0", self.port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PAC 모니터</title>
<style>
:root{--bg:#f4f5f7;--card:#ffffff;--ink:#1d2129;--muted:#6b7280;--line:#e3e5e8;--ok:#16a34a;--fail:#dc2626;
--run:#2563eb;--warn:#d97706;--pend:#9ca3af}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--card:#1a1d24;--ink:#e8eaee;--muted:#9aa1ad;--line:#2a2f38}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,"Noto Sans KR",sans-serif}
header{display:flex;flex-wrap:wrap;align-items:center;gap:12px 20px;padding:12px 16px;background:var(--card);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:2}
h1{font-size:17px;margin:0}.pill{padding:3px 10px;border-radius:99px;font-weight:600;font-size:13px;color:#fff;background:var(--pend)}
.pill.running{background:var(--run)}.pill.done{background:var(--ok)}.pill.error,.pill.stopped{background:var(--fail)}
.kv{color:var(--muted)}.kv b{color:var(--ink)}#stop{margin-left:auto;background:var(--fail);color:#fff;border:0;border-radius:8px;padding:9px 18px;font-weight:700;font-size:15px;cursor:pointer}
main{display:grid;gap:14px;padding:14px 16px;grid-template-columns:minmax(0,2fr) minmax(280px,1fr)}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;min-width:0}
.card h2{font-size:14px;margin:0 0 8px}.card img{width:100%;border-radius:8px;display:block;background:#000}
.cap{color:var(--muted);font-size:12.5px;margin-top:6px;white-space:pre-wrap}
.steps{list-style:none;margin:0;padding:0}.steps li{display:flex;align-items:center;gap:10px;padding:7px 4px;border-bottom:1px solid var(--line)}
.dot{width:12px;height:12px;border-radius:50%;background:var(--pend);flex:none}.running .dot{background:var(--run);box-shadow:0 0 0 4px #2563eb33}
.ok .dot,.done .dot{background:var(--ok)}.fail .dot{background:var(--fail)}.cnt{margin-left:auto;color:var(--muted);font-size:12px}
.note{color:var(--muted);font-size:12px}.metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:10px}
.m{background:var(--bg);border-radius:8px;padding:8px}.m .k{color:var(--muted);font-size:12px}.m .v{font-size:18px;font-weight:700}
.panels{grid-column:1/-1;display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(330px,1fr))}
#log{grid-column:1/2;max-height:320px;overflow:auto;font:12.5px/1.5 ui-monospace,monospace}
#log div{padding:1px 0;border-bottom:1px dotted var(--line)}#log .warn{color:var(--warn)}#log .error{color:var(--fail);font-weight:700}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:5px 4px;border-bottom:1px solid var(--line);text-align:left}
.timing td:last-child{text-align:right;font-variant-numeric:tabular-nums}
</style></head><body>
<header><h1>PAC 2026 · 파티션 적재 모니터</h1><span id="status" class="pill">idle</span>
<span class="kv">실행 <b id="run">-</b></span><span class="kv">로봇 <b id="robot">-</b></span>
<span class="kv">카메라 <b id="cam">-</b></span><span class="kv">적재 <b id="placed">0</b></span>
<span class="kv">경과 <b id="elapsed">-</b></span><button id="stop">■ 비상 정지</button></header>
<main>
 <section class="card"><h2>실시간 화면 (인식 결과 오버레이)</h2><img src="/live.mjpg" alt="live camera">
  <div class="cap">빨강: 상자 · 칸 색: 회색 빈 칸 / 초록 정상 / 빨강 걸침 / 주황 불확실 · 굵은 칸: 목표 칸 · 보라: 블록, 빨간 선: 그리퍼 닫는 방향</div></section>
 <section class="card"><h2>단계 진행</h2><ul class="steps" id="steps"></ul>
  <div class="metrics" id="metrics"></div></section>
 <div class="panels" id="panels"></div>
 <section class="card" id="logc"><h2>실시간 로그</h2><div id="log"></div></section>
 <section class="card"><h2>모델 처리 시간</h2><table class="timing" id="timing"></table>
  <h2 style="margin-top:14px">지난 실행</h2><table id="hist"></table></section>
</main>
<script>
const $=id=>document.getElementById(id);let pv={},lastLog=0;
$("stop").onclick=()=>{if(confirm("로봇을 정지할까요?"))fetch("/stop",{method:"POST"})};
const M={box_height_cm:"상자 높이 (cm)",height_vggt_cm:"VGGT 높이 (cm)",grid:"칸 n×m",empty_cells:"빈 칸",
 blocks_supply:"공급 영역 블록",gripper_mm:"그리퍼 폭 (mm)",target_cell:"목표 칸",retries:"재시도"};
async function tick(){try{const s=await (await fetch("/state")).json();
 $("status").textContent=s.status;$("status").className="pill "+s.status;$("run").textContent=s.run_id||"-";
 $("robot").textContent=s.info.robot||"-";$("cam").textContent=s.info.camera||"-";
 $("placed").textContent=(s.metrics.placed??0)+" / "+(s.info.max_blocks??"-");$("elapsed").textContent=s.elapsed!=null?s.elapsed+" s":"-";
 $("steps").innerHTML=Object.entries(s.step_names).map(([n,name])=>{const st=s.steps[n]||{status:"pending",ok:0,fail:0,note:""};
  return `<li class="${st.status}"><span class="dot"></span><div><b>${n}. ${name}</b><div class="note">${st.note||""}</div></div>
  <span class="cnt">✓${st.ok} ✗${st.fail}</span></li>`}).join("");
 $("metrics").innerHTML=Object.entries(M).filter(([k])=>s.metrics[k]!==undefined).map(([k,l])=>`<div class="m"><div class="k">${l}</div><div class="v">${s.metrics[k]}</div></div>`).join("");
 const p=$("panels");for(const [k,title] of Object.entries(s.panel_titles)){const info=s.panels[k];if(!info)continue;
  let c=$("p_"+k);if(!c){c=document.createElement("section");c.className="card";c.id="p_"+k;c.innerHTML=`<h2>${title}</h2><img><div class="cap"></div>`;p.appendChild(c)}
  if(pv[k]!==info.v){c.querySelector("img").src=`/panel/${k}.jpg?v=${info.v}`;pv[k]=info.v}c.querySelector(".cap").textContent=info.caption||""}
 const L=$("log"),atBottom=L.scrollTop+L.clientHeight>=L.scrollHeight-20;
 L.innerHTML=s.logs.map(l=>`<div class="${l.level}">[${l.t.toFixed(1)}s] ${l.msg.replace(/</g,"&lt;")}</div>`).join("");if(atBottom)L.scrollTop=L.scrollHeight;
 const t=s.metrics.timings_ms||{};$("timing").innerHTML=Object.entries(t).map(([k,v])=>`<tr><td>${k}</td><td>${v} ms</td></tr>`).join("")||"<tr><td>-</td></tr>";
 $("hist").innerHTML="<tr><th>실행</th><th>적재</th><th>격자</th><th>비고</th></tr>"+s.history.map(h=>`<tr><td>${h.run}</td><td>${h.placed??"-"}</td><td>${h.grid??"-"}</td><td>${h.stopped??""}</td></tr>`).join("");
}catch(e){$("status").textContent="연결 끊김";$("status").className="pill error"}}
setInterval(tick,500);tick();
</script></body></html>"""
