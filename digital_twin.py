#!/usr/bin/env python3
"""
Digital Twin Pipeline
=====================
Phase 1 — The Eyes      : YOLOv8 + ByteTrack  → per-frame vehicle detections
Phase 2 — The Translator: Homography matrix    → pixels → SUMO world coords
Phase 3 — The Controller: SUMO TraCI           → spawn / move / remove twins

Usage:
  python3 digital_twin.py                 # full pipeline (SUMO + video)
  python3 digital_twin.py --no-sumo       # video + coordinate overlay only
  python3 digital_twin.py --calibrate     # interactive homography point-picker
"""

import cv2
import numpy as np
import torch
import sys
import os
import subprocess
import time
import argparse
import json

# ── Windows Per-Monitor DPI awareness (prevents blurry window on 125%/150% displays) ──
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # 2 = Per-Monitor DPI aware
    except Exception:
        pass  # older Windows versions may not support this

from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
VIDEO_PATH      = "./videos/video3.mp4"
MODEL_PATH      = "./models/yolov8n.pt"
SUMO_CFG        = "my_config.sumocfg"
TRACI_PORT      = 8813
VEHICLE_TYPE    = "standard_car"
DETECT_CLASSES  = [1, 2, 3, 5, 7]   # COCO: car, motorcycle, bus, truck
SKIP_FRAMES     = 2               # process every Nth frame
PROCESS_SCALE    = 0.5              # resize factor for YOLO inference (↓ = faster, 1.0 = full res)
DISPLAY_SCALE    = 0.3             # resize factor for the display window (↓ = smaller window)

# ── HYBRID TWIN CONFIG ────────────────────────────────────────────────────────
TRIPWIRE_MARGIN   = 0.10            # fraction from each edge that acts as entry tripwire
TL_JUNCTION_ID    = "J25"           # SUMO junction / TL node id at the intersection
TL_PHASE_NS_GREEN = 0               # SUMO phase index where North-South has green
TL_PHASE_EW_GREEN = 2               # SUMO phase index where East-West has green
STOP_PX_THRESHOLD = 5.0             # px/frame below which a vehicle is considered stopped
STOP_MIN_VEHICLES = 1               # min stopped vehicles on an arm to infer red light

# ── HOMOGRAPHY CALIBRATION POINTS ─────────────────────────────────────────────
# Map 4 known pixel positions  →  their corresponding SUMO (x, y) coordinates.
# SUMO network dead-ends (from crossroads.net.xml):
#   J24=West(-61,-27)  J26=North(-11,18)  J27=East(33,-26)  J28=South(-10,-66)
#
# Default pixel values are *estimates* for a typical overhead crossroads camera.
# Run `python3 digital_twin.py --calibrate`  to click your own exact values.

PIXEL_PTS = np.float32([
    [3, 591],
    [966, 6],
    [1914, 750],
    [236, 1076],
])

SUMO_PTS = np.float32([
    [-61.00, -27.22],   # J24
    [-10.77,  18.31],   # J26
    [ 32.66, -26.23],   # J27
    [-10.15, -66.07],   # J28
])


# ── PHASE 2: HOMOGRAPHY MAPPER ────────────────────────────────────────────────
class HomographyMapper:
    """Bidirectional mapping between video pixels and SUMO world coordinates."""

    def __init__(self, pixel_pts: np.ndarray, sumo_pts: np.ndarray):
        self.H,     _ = cv2.findHomography(pixel_pts, sumo_pts)
        self.H_inv, _ = cv2.findHomography(sumo_pts,  pixel_pts)

    def pixel_to_sumo(self, px: float, py: float):
        """(pixel_x, pixel_y) → (sumo_x, sumo_y)"""
        pt     = np.array([[[px, py]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.H)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    def sumo_to_pixel(self, sx: float, sy: float):
        """(sumo_x, sumo_y) → (pixel_x, pixel_y) for overlay drawing."""
        pt     = np.array([[[sx, sy]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.H_inv)
        return int(mapped[0, 0, 0]), int(mapped[0, 0, 1])


# ── ZONE DETECTION (pixel frame → SUMO road arm) ─────────────────────────────
# The video frame is split into 5 zones.  The outer 28 % on each side maps to
# the road arm that enters from that direction.
#
#   North zone  → vehicles travelling on edge E17  (J26 → J25)
#   South zone  → vehicles travelling on edge E19  (J28 → J25)
#   West  zone  → vehicles travelling on edge E16  (J24 → J25)
#   East  zone  → vehicles travelling on edge E18  (J27 → J25)
#   Center zone → intersection box  (use N-S road as fallback)

ZONE_MARGIN = 0.40   # outer fraction of frame considered a "road arm" zone
                     # raise this if camera is angled (not perfectly top-down)

# Maps zone name → (traci_route_id,  [ordered edges for that route])
ZONE_ROUTE: dict = {
    "north":  ("twin_NS", ["E17", "-E19"]),
    "south":  ("twin_SN", ["E19", "-E17"]),
    "west":   ("twin_WE", ["E16", "-E18"]),
    "east":   ("twin_EW", ["E18", "-E16"]),
    "center": ("twin_NS", ["E17", "-E19"]),  # fallback
}

# Maps zone → the primary SUMO edge for that arm (zone classification only)
ZONE_EDGE: dict = {
    "north": "E17",
    "south": "E19",
    "west":  "E16",
    "east":  "E18",
    "center": "",
}


def detect_zone(cx: int, cy: int, width: int, height: int) -> str:
    """Classify a pixel centroid into a road-arm zone."""
    rx, ry = cx / width, cy / height  # normalised [0, 1]
    if ry < ZONE_MARGIN:        return "north"
    if ry > 1 - ZONE_MARGIN:   return "south"
    if rx < ZONE_MARGIN:        return "west"
    if rx > 1 - ZONE_MARGIN:   return "east"
    return "center"


# ── PHASE 3: HYBRID TRACI TWIN CONTROLLER ────────────────────────────────────
class HybridTwinController:
    """
    Hybrid Autonomous Digital Twin Controller.

    Spawn strategy
    --------------
    • Frame-0 snapshot  : All vehicles visible on the very first processed frame
      are spawned immediately in SUMO (Rule 1).
    • Edge tripwires    : Any new track_id whose centroid falls within
      TRIPWIRE_MARGIN of a frame edge triggers a spawn, because the vehicle
      is entering the scene from outside (Rule 3).
    • No continuous coordinate mirroring — vehicles drive autonomously under
      SUMO's car-following model after their initial spawn (Rule 4).

    Traffic-light sync (Rule 2)
    ---------------------------
    Per-arm movement is estimated each frame. If ≥ STOP_MIN_VEHICLES vehicles
    on an arm all have speed < STOP_PX_THRESHOLD px/frame, that arm is declared
    "stopped" (inferred red). The SUMO TL phase is set to give green to the
    opposing moving arm and held until the next sync event.
    """

    def __init__(self, cfg_path: str, port: int = TRACI_PORT,
                 fps: float = 30.0, skip_frames: int = SKIP_FRAMES,
                 tripinfo_output: str = "", skip_tl_sync: bool = False):
        import traci
        self._traci = traci

        binary = "sumo-gui" if self._binary_exists("sumo-gui") else "sumo"
        cmd = [
            binary, "-c", cfg_path,
            "--remote-port", str(port),
            "--start",
            "--quit-on-end", "false",
            "--no-step-log",
        ]
        if tripinfo_output:
            cmd += ["--tripinfo-output", tripinfo_output]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(2.0)
        traci.init(port)

        # Pre-create one route per road arm
        seen_routes: set = set()
        for route_id, edges in ZONE_ROUTE.values():
            if route_id not in seen_routes:
                try:
                    traci.route.add(route_id, edges)
                except Exception:
                    pass
                seen_routes.add(route_id)

        self._active: dict[int, str]   = {}   # track_id → SUMO vehicle id
        self._ever_seen: set[int]      = set() # never re-spawn the same id
        self._prev_centroids: dict[int, tuple] = {}  # for speed estimation
        self._tl_phase: str            = "unknown"   # last inferred TL state
        self._step                     = 0

        # Wait-time tracking
        self._spf: float               = skip_frames / fps  # seconds per processed frame
        self._wait_frames: dict[int, int] = {}   # track_id → consecutive stopped frames
        self._zone_by_tid: dict[int, str] = {}   # track_id → current zone

        self._skip_tl_sync = skip_tl_sync

        print(f"[✓] SUMO started ({binary}) on port {port}")
        print("[✓] Hybrid mode: Frame-0 snapshot + edge tripwires" +
              (" + AI TL control" if skip_tl_sync else " + TL sync"))

    @staticmethod
    def _binary_exists(name: str) -> bool:
        return subprocess.run(["where" if sys.platform == "win32" else "which", name],
                              capture_output=True).returncode == 0

    # ── Shared spawn helper ───────────────────────────────────────────────────
    def _spawn(self, tid: int, zone: str):
        """Spawn a SUMO twin once; silently no-ops if already spawned."""
        if tid in self._ever_seen:
            return
        self._ever_seen.add(tid)
        vid      = f"twin_{tid}"
        route_id, _ = ZONE_ROUTE.get(zone, ZONE_ROUTE["center"])
        try:
            self._traci.vehicle.add(
                vid, route_id,
                typeID=VEHICLE_TYPE,
                departLane="best",
                departPos="base",
                departSpeed="max",
            )
            self._active[tid] = vid
            print(f"  [+] Spawned twin_{tid} on {route_id}  (zone: {zone})")
        except Exception as exc:
            print(f"  [!] Could not spawn twin_{tid}: {exc}")

    # ── Rule 1: Frame-0 snapshot ──────────────────────────────────────────────
    def init_frame0(self, detections: list):
        """Spawn every vehicle visible on the very first processed frame."""
        print(f"[Frame-0] Snapshot: {len(detections)} vehicles detected — spawning all.")
        for (tid, cx, cy, zone) in detections:
            self._spawn(tid, zone)
            self._prev_centroids[tid] = (cx, cy)
            self._wait_frames[tid]    = 0
            self._zone_by_tid[tid]    = zone
        self._traci.simulationStep()
        self._step += 1

    # ── Rule 3: Edge tripwire check ───────────────────────────────────────────
    def _is_tripwire_entry(self, cx: int, cy: int, width: int, height: int) -> bool:
        """True if the centroid is inside the border tripwire margin zone."""
        rx, ry = cx / width, cy / height
        return (
            ry < TRIPWIRE_MARGIN
            or ry > 1.0 - TRIPWIRE_MARGIN
            or rx < TRIPWIRE_MARGIN
            or rx > 1.0 - TRIPWIRE_MARGIN
        )

    # ── Rule 2: Traffic-light synchronisation ────────────────────────────────
    def _sync_traffic_light(self, arm_speeds: dict):
        """
        Infer real-world TL state from per-arm vehicle speeds and mirror in SUMO.
        arm_speeds: {zone_name: [speed_px_per_frame, ...]}
        """
        def arm_stopped(arm: str) -> bool:
            speeds  = arm_speeds.get(arm, [])
            stopped = sum(1 for s in speeds if s < STOP_PX_THRESHOLD)
            return stopped >= STOP_MIN_VEHICLES

        ns_stopped = arm_stopped("north") or arm_stopped("south")
        ew_stopped = arm_stopped("east")  or arm_stopped("west")

        if   ns_stopped and not ew_stopped:
            desired = "EW_GREEN"   # NS queuing → give EW the green
        elif ew_stopped and not ns_stopped:
            desired = "NS_GREEN"   # EW queuing → give NS the green
        else:
            return   # ambiguous — leave TL as-is

        if desired == self._tl_phase:
            return   # already correct, no TraCI call needed

        self._tl_phase = desired
        phase_idx = TL_PHASE_NS_GREEN if desired == "NS_GREEN" else TL_PHASE_EW_GREEN
        try:
            self._traci.trafficlight.setPhase(TL_JUNCTION_ID, phase_idx)
            self._traci.trafficlight.setPhaseDuration(TL_JUNCTION_ID, 9999)
            print(f"  [TL] ▶ {desired}  (phase index {phase_idx})")
        except Exception:
            pass   # TL node may not exist in all networks — fail gracefully

    # ── Per-frame tick (Rules 2, 3, 4) ───────────────────────────────────────
    def tick(self, detections: list, width: int, height: int):
        """
        Called once per processed video frame (after frame-0).
        detections: [(track_id, cx_pixel, cy_pixel, zone), ...]
        """
        traci    = self._traci
        seen_ids: set[int]       = set()
        arm_speeds: dict         = {z: [] for z in ZONE_ROUTE}

        for (tid, cx, cy, zone) in detections:
            seen_ids.add(tid)
            self._zone_by_tid[tid] = zone

            # Rule 3 — tripwire spawn for genuinely new vehicles at the edge
            if tid not in self._ever_seen:
                if self._is_tripwire_entry(cx, cy, width, height):
                    self._spawn(tid, zone)

            # Rule 2 — accumulate per-arm speeds for TL inference
            speed = 0.0
            if tid in self._prev_centroids:
                px, py = self._prev_centroids[tid]
                speed  = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                arm_speeds[zone].append(speed)

            self._prev_centroids[tid] = (cx, cy)

            # Wait-time tracking
            if speed < STOP_PX_THRESHOLD:
                self._wait_frames[tid] = self._wait_frames.get(tid, 0) + 1
            else:
                self._wait_frames[tid] = 0

        # Rule 2 - sync SUMO traffic light (skip when AI controller handles TL)
        if not self._skip_tl_sync:
            self._sync_traffic_light(arm_speeds)

        # Rule 4 — remove SUMO twins for tracks that left the video
        gone = set(self._active.keys()) - seen_ids
        for tid in gone:
            vid = self._active.pop(tid)
            try:
                # Only remove if vehicle still exists in SUMO
                if vid in traci.vehicle.getIDList():
                    traci.vehicle.remove(vid)
                print(f"  [-] Removed twin_{tid}")
            except Exception:
                pass
            self._prev_centroids.pop(tid, None)
            self._wait_frames.pop(tid, None)
            self._zone_by_tid.pop(tid, None)

        traci.simulationStep()
        self._step += 1

    def spawned_count(self) -> int:
        return len(self._ever_seen)

    def active_count(self) -> int:
        return len(self._active)

    def tl_phase(self) -> str:
        return self._tl_phase

    def get_wait_stats(self) -> dict:
        """
        Returns max wait time in seconds per arm, plus overall max.
        Keys: 'north', 'south', 'east', 'west', 'max'
        """
        arm_waits: dict[str, list] = {a: [] for a in ("north", "south", "east", "west")}
        for tid, frames in self._wait_frames.items():
            zone = self._zone_by_tid.get(tid, "center")
            if zone in arm_waits:
                arm_waits[zone].append(frames * self._spf)
        result = {arm: (max(waits) if waits else 0.0) for arm, waits in arm_waits.items()}
        result["max"] = max(result.values()) if result else 0.0
        return result

    def close(self):
        try:
            self._traci.close()
        except Exception:
            pass
        self._proc.terminate()
        print("[✓] SUMO connection closed.")


# ── CALIBRATION HELPER ────────────────────────────────────────────────────────
def run_calibration(video_path: str):
    """Click 4 road dead-end points on the first video frame → print PIXEL_PTS."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("[✗] Cannot open video for calibration.")
        return

    labels = [
        "1. West dead-end  (J24)  → SUMO (-61, -27)",
        "2. North dead-end (J26)  → SUMO (-11,  18)",
        "3. East dead-end  (J27)  → SUMO ( 33, -26)",
        "4. South dead-end (J28)  → SUMO (-10, -66)",
    ]
    pts   = []
    canvas = frame.copy()

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(canvas, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(canvas, labels[len(pts) - 1].split("→")[0].strip(),
                        (x + 10, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Calibration - click 4 road ends", canvas)
            print(f"  Registered: {labels[len(pts)-1]}  →  pixel ({x}, {y})")
            if len(pts) == 4:
                print("\n[✓] Copy these PIXEL_PTS into digital_twin.py:\n")
                print("PIXEL_PTS = np.float32([")
                for p in pts:
                    print(f"    [{p[0]}, {p[1]}],")
                print("])")

    print("\n=== CALIBRATION MODE ===")
    print("Click the 4 dead-end tips of each road arm in this order:")
    for l in labels:
        print(" ", l)
    print("\nPress Q to exit.\n")

    cv2.namedWindow("Calibration - click 4 road ends", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration - click 4 road ends", 1280, 720)  # explicit size so window is visible
    cv2.imshow("Calibration - click 4 road ends", canvas)
    cv2.waitKey(1)   # pump the event loop so Windows creates the actual HWND
    cv2.setMouseCallback("Calibration - click 4 road ends", on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hybrid Autonomous Digital Twin")
    parser.add_argument("--video",        default=VIDEO_PATH,  help="Path to input video")
    parser.add_argument("--port",         type=int, default=TRACI_PORT, help="TraCI port for SUMO")
    parser.add_argument("--tripinfo",     default="", help="Path for SUMO tripinfo XML output")
    parser.add_argument("--window-title", default="Digital Twin - Hybrid Mode", help="OpenCV window title")
    parser.add_argument("--summary-json", default="", help="Path to write summary JSON at exit")
    parser.add_argument("--no-sumo",      action="store_true", help="Video-only mode (no SUMO)")
    parser.add_argument("--calibrate",    action="store_true", help="Homography calibration mode")
    parser.add_argument("--use-ai",       action="store_true", help="Use MAPPO AI for TL control")
    parser.add_argument("--ai-checkpoint", default="models/mappo_checkpoint.pt", help="AI model checkpoint")
    parser.add_argument("--fixed-tl",     action="store_true", help="Use SUMO default fixed-timer TL (baseline)")
    args = parser.parse_args()

    if args.calibrate:
        run_calibration(args.video)
        return

    video_path    = args.video
    traci_port    = args.port
    tripinfo_out  = args.tripinfo
    win_title     = args.window_title
    summary_path  = args.summary_json
    no_sumo       = args.no_sumo
    use_ai        = args.use_ai
    ai_checkpoint = args.ai_checkpoint
    fixed_tl      = args.fixed_tl
    # skip_tl_sync if AI controls TL or if we want SUMO default fixed timers
    skip_tl       = use_ai or fixed_tl

    # Check TraCI availability
    traci_ok = False
    try:
        import traci  # noqa: just checking import
        traci_ok = True
    except ImportError:
        print("[!] traci not found → SUMO integration disabled.")

    # Decide whether to use SUMO
    use_sumo = traci_ok and not no_sumo

    # ── Load YOLO model ──────────────────────────────────────────────────────
    print(f"[*] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    if torch.cuda.is_available():
        model.to("cuda")
        print("[✓] Using CUDA GPU")
    else:
        print("[!] CUDA unavailable — running on CPU (slower)")

    # ── Open video ───────────────────────────────────────────────────────────
    cap    = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print(f"[*] Video: {width}x{height}  @{fps:.1f} fps")

    # Compute reduced inference resolution
    proc_w = int(width  * PROCESS_SCALE)
    proc_h = int(height * PROCESS_SCALE)
    print(f"[*] Inference resolution: {proc_w}x{proc_h}  (scale={PROCESS_SCALE})")

    # ── Phase 2: build Homography ────────────────────────────────────────────
    mapper = HomographyMapper(PIXEL_PTS, SUMO_PTS)
    print("[✓] Homography matrix computed")

    # ── Phase 3: start SUMO (HybridTwinController) ───────────────────────────
    sumo_ctrl = None
    if use_sumo:
        try:
            sumo_ctrl = HybridTwinController(
                SUMO_CFG, port=traci_port, fps=fps,
                skip_frames=SKIP_FRAMES, tripinfo_output=tripinfo_out,
                skip_tl_sync=skip_tl
            )
        except Exception as exc:
            print(f"[!] SUMO failed to start: {exc}")
            print("    -> Continuing in video-only mode.")
            print("    -> Install SUMO with:  sudo apt install sumo sumo-tools")

    # ── Optional: AI Traffic Controller ───────────────────────────────────────
    ai_ctrl = None
    if use_ai and sumo_ctrl:
        try:
            import traci as traci_mod
            from ai_controller import AITrafficController
            ai_ctrl = AITrafficController(traci_mod, checkpoint=ai_checkpoint,
                                          step_length=SKIP_FRAMES / fps)
            print("[+] AI Traffic Controller active (MAPPO + Safety + Preemption)")
        except Exception as exc:
            print(f"[!] AI Controller failed to load: {exc}")
            print("    -> Falling back to heuristic TL sync.")

    # Precompute tripwire pixel positions for overlay drawing
    tw_top    = int(height * TRIPWIRE_MARGIN)
    tw_bottom = int(height * (1 - TRIPWIRE_MARGIN))
    tw_left   = int(width  * TRIPWIRE_MARGIN)
    tw_right  = int(width  * (1 - TRIPWIRE_MARGIN))

    # ── Main loop ────────────────────────────────────────────────────────────
    frame_count   = 0
    frame0_done   = False   # Rule 1: have we done the frame-0 snapshot yet?
    print("\n[*] Hybrid pipeline running — press Q to quit.\n")

    # Pre-create the display window so it reliably appears on Windows
    win_w = int(width  * DISPLAY_SCALE)
    win_h = int(height * DISPLAY_SCALE)
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, win_w, win_h)
    cv2.waitKey(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue

        # ── PHASE 1: Detection & Tracking ────────────────────────────────────
        proc_frame = cv2.resize(frame, (proc_w, proc_h))
        results = model.track(
            proc_frame,
            persist=True,
            conf=0.25,
            imgsz=max(proc_w, proc_h),
            classes=DETECT_CLASSES,
            verbose=False,
        )

        detections = []   # [(track_id, cx, cy, zone), ...]

        if results[0].boxes.id is not None:
            boxes     = results[0].boxes.xyxy.cpu().numpy() / PROCESS_SCALE
            track_ids = results[0].boxes.id.int().cpu().numpy()
            classes   = results[0].boxes.cls.int().cpu().numpy()
            cls_names = results[0].names

            for box, tid, cls_id in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = box
                cx   = int((x1 + x2) / 2)
                cy   = int((y1 + y2) / 2)
                zone = detect_zone(cx, cy, width, height)
                detections.append((int(tid), cx, cy, zone))

                # ── PHASE 2 overlay ──────────────────────────────────────────
                sx, sy = mapper.pixel_to_sumo(cx, cy)
                label  = f"#{tid} {cls_names[cls_id]} ({sx:.0f},{sy:.0f}) [{zone[0].upper()}]"

                zone_colours = {
                    "north": (0, 200, 255),
                    "south": (0, 100, 255),
                    "east":  (255, 200, 0),
                    "west":  (255, 80,  0),
                    "center":(200, 200, 200),
                }
                colour = zone_colours.get(zone, (255, 255, 255))

                cv2.rectangle(frame,
                              (int(x1), int(y1)), (int(x2), int(y2)),
                              colour, 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(frame, label,
                            (int(x1), max(int(y1) - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (0, 255, 255), 1)

        # ── PHASE 3: Hybrid TraCI sync ────────────────────────────────────────
        if sumo_ctrl:
            if not frame0_done:
                # Rule 1 -- Frame-0 snapshot: spawn everything visible right now
                sumo_ctrl.init_frame0(detections)
                frame0_done = True
            else:
                # Rules 2, 3, 4 -- tripwire + TL sync + autonomous driving
                sumo_ctrl.tick(detections, width, height)

            # AI controller overrides heuristic TL sync each step
            if ai_ctrl:
                ai_ctrl.step()

        # ── HUD overlays ──────────────────────────────────────────────────────
        # Draw homography anchor points (red crosses)
        for (px, py) in PIXEL_PTS.astype(int):
            cv2.drawMarker(frame, (px, py), (0, 0, 255),
                           cv2.MARKER_CROSS, 22, 2)

        # Draw 4 cyan tripwire lines near frame edges (Rule 3 visual)
        tw_colour = (255, 255, 0)   # cyan
        cv2.line(frame, (0, tw_top),    (width, tw_top),    tw_colour, 1)  # North
        cv2.line(frame, (0, tw_bottom), (width, tw_bottom), tw_colour, 1)  # South
        cv2.line(frame, (tw_left,  0),  (tw_left,  height), tw_colour, 1)  # West
        cv2.line(frame, (tw_right, 0),  (tw_right, height), tw_colour, 1)  # East

        if sumo_ctrl:
            tl_str      = sumo_ctrl.tl_phase()
            ws          = sumo_ctrl.get_wait_stats()

            if ai_ctrl:
                ai_stats    = ai_ctrl.get_stats()
                mode_text   = "AI [MAPPO + Safety + Preemption]"
                sumo_counts = (f"Tracked: {len(detections)}  |  "
                               f"SUMO active: {sumo_ctrl.active_count()}  |  "
                               f"Phase: {ai_stats['current_phase']}({ai_stats['state']})  |  "
                               f"Switches: {ai_stats['phase_switches']}")
            else:
                mode_text   = "HYBRID [Frame0 + Tripwire + TL-Sync + Wait-Time]"
                sumo_counts = (f"Tracked: {len(detections)}  |  "
                               f"SUMO active: {sumo_ctrl.active_count()}  |  "
                               f"Spawned: {sumo_ctrl.spawned_count()}  |  "
                               f"TL: {tl_str}")
            wait_line   = (f"Wait: N={ws['north']:.0f}s  "
                           f"S={ws['south']:.0f}s  "
                           f"E={ws['east']:.0f}s  "
                           f"W={ws['west']:.0f}s  "
                           f"| Max={ws['max']:.0f}s")
        else:
            mode_text   = "PHASE 1+2 [VIDEO ONLY — no SUMO]"
            sumo_counts = f"Tracked: {len(detections)} vehicles"
            wait_line   = ""

        cv2.rectangle(frame, (0, 0), (width, 72), (0, 0, 0), -1)
        cv2.putText(frame, mode_text,  (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
        cv2.putText(frame, sumo_counts, (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
        cv2.putText(frame, wait_line,  (8, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1)

        display_frame = cv2.resize(frame, (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)))
        cv2.imshow(win_title, display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()

    # Write summary JSON if requested
    if summary_path and sumo_ctrl:
        ws = sumo_ctrl.get_wait_stats()
        summary = {
            "frames_processed": frame_count // SKIP_FRAMES,
            "total_spawned":    sumo_ctrl.spawned_count(),
            "wait_stats":       ws,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[✓] Summary written to {summary_path}")

    if sumo_ctrl:
        sumo_ctrl.close()
    cv2.destroyAllWindows()
    print(f"[✓] Finished. Processed {frame_count // SKIP_FRAMES} frames.")


if __name__ == "__main__":
    main()
