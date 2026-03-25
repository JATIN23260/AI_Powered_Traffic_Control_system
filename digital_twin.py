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

from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
VIDEO_PATH      = "crossroad.mp4"
MODEL_PATH      = "yolov8l.pt"
SUMO_CFG        = "my_config.sumocfg"
TRACI_PORT      = 8813
VEHICLE_TYPE    = "standard_car"
DETECT_CLASSES  = [2, 3, 5, 7]   # COCO: car, motorcycle, bus, truck
SKIP_FRAMES     = 2               # process every Nth frame

# ── HOMOGRAPHY CALIBRATION POINTS ─────────────────────────────────────────────
# Map 4 known pixel positions  →  their corresponding SUMO (x, y) coordinates.
# SUMO network dead-ends (from crossroads.net.xml):
#   J24=West(-61,-27)  J26=North(-11,18)  J27=East(33,-26)  J28=South(-10,-66)
#
# Default pixel values are *estimates* for a typical overhead crossroads camera.
# Run `python3 digital_twin.py --calibrate`  to click your own exact values.

PIXEL_PTS = np.float32([
    [  60, 370],   # West  road end  → J24
    [ 640,  35],   # North road end  → J26
    [1220, 360],   # East  road end  → J27
    [ 640, 685],   # South road end  → J28
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

ZONE_MARGIN = 0.28   # outer fraction of frame considered a "road arm" zone

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


# ── PHASE 3: TRACI TWIN CONTROLLER ───────────────────────────────────────────
class SumoTwinController:
    """
    Manages digital-twin vehicles in a live SUMO simulation via TraCI.

    Spawn-only mode
    ---------------
    • When a vehicle is first detected in the video (including every vehicle
      visible at the very first processed frame), it is spawned once in SUMO
      on the road-arm edge that matches its pixel zone.
    • After spawning, the SUMO vehicle drives **autonomously** — no coordinate
      mirroring (moveToXY) is performed at any point.
    • When the video tracker loses a vehicle the matching SUMO twin is removed
      so the simulation stays clean.
    • A track_id that re-appears (tracker re-assigns the same id) is NOT
      re-spawned because it is remembered in _ever_seen.
    """

    def __init__(self, cfg_path: str, port: int = TRACI_PORT):
        import traci
        self._traci = traci

        # Pick sumo-gui if available, otherwise headless sumo
        binary = "sumo-gui" if self._binary_exists("sumo-gui") else "sumo"
        cmd = [
            binary,
            "-c", cfg_path,
            "--remote-port", str(port),
            "--start",                    # start simulation immediately
            "--quit-on-end", "false",     # keep SUMO open after video ends
            "--no-step-log",              # quieter console output
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(2.0)   # give SUMO time to bind the port
        traci.init(port)

        # Pre-create one route per road arm so twins spawn on the correct edge
        seen_routes: set = set()
        for route_id, edges in ZONE_ROUTE.values():
            if route_id not in seen_routes:
                try:
                    traci.route.add(route_id, edges)
                except Exception:
                    pass
                seen_routes.add(route_id)

        self._active: dict[int, str] = {}  # track_id → SUMO vehicle id (currently in sim)
        self._ever_seen: set[int]    = set()  # all track_ids ever spawned (no re-spawn)
        self._step = 0
        print(f"[✓] SUMO started ({binary}) on port {port}")
        print("[✓] Spawn-only mode: vehicles drive autonomously after initial spawn")

    @staticmethod
    def _binary_exists(name: str) -> bool:
        return subprocess.run(
            ["which", name], capture_output=True
        ).returncode == 0

    def tick(self, detections: list, mapper: HomographyMapper):
        """
        Called once per processed video frame.
        detections: [(track_id, cx_pixel, cy_pixel, zone), ...]
          zone: one of 'north','south','east','west','center'

        Behaviour:
          - First time a track_id is seen → spawn a SUMO twin (one-time only).
          - Already-spawned vehicles → do nothing (autonomous SUMO driving).
          - Disappeared tracks → remove from SUMO.
        """
        traci = self._traci
        seen_ids: set[int] = set()

        for (tid, cx, cy, zone) in detections:
            vid      = f"twin_{tid}"
            route_id, _ = ZONE_ROUTE[zone]
            seen_ids.add(tid)

            # ── Spawn once, the very first time this track_id appears ──────────
            if tid not in self._ever_seen:
                self._ever_seen.add(tid)
                try:
                    traci.vehicle.add(
                        vid, route_id,
                        typeID=VEHICLE_TYPE,
                        departLane="best",
                        departPos="base",
                        departSpeed="0",
                    )
                    self._active[tid] = vid
                    print(f"  [+] Spawned twin_{tid} on route {route_id} (zone: {zone})")
                except Exception as exc:
                    # Vehicle may already exist (e.g. SUMO pre-loaded it) — ignore
                    print(f"  [!] Could not spawn twin_{tid}: {exc}")

            # NOTE: No moveToXY — SUMO vehicle drives autonomously from here.

        # ── Remove twins for tracks that have disappeared from the video ───────
        gone = set(self._active.keys()) - seen_ids
        for tid in gone:
            vid = self._active.pop(tid)
            try:
                traci.vehicle.remove(vid)
                print(f"  [-] Removed twin_{tid} (no longer in video)")
            except Exception:
                pass   # already gone in SUMO

        traci.simulationStep()
        self._step += 1

    def spawned_count(self) -> int:
        """Total number of unique vehicles ever spawned in this session."""
        return len(self._ever_seen)

    def active_count(self) -> int:
        """Number of SUMO twins currently in the simulation."""
        return len(self._active)

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
            cv2.imshow("Calibration — click 4 road ends", canvas)
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

    cv2.imshow("Calibration — click 4 road ends", canvas)
    cv2.setMouseCallback("Calibration — click 4 road ends", on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    do_calibrate = "--calibrate" in sys.argv
    no_sumo      = "--no-sumo"   in sys.argv

    if do_calibrate:
        run_calibration(VIDEO_PATH)
        return

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
    cap    = cv2.VideoCapture(VIDEO_PATH)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print(f"[*] Video: {width}x{height}  @{fps:.1f} fps")

    # ── Phase 2: build Homography ────────────────────────────────────────────
    mapper = HomographyMapper(PIXEL_PTS, SUMO_PTS)
    print("[✓] Homography matrix computed")

    # ── Phase 3: start SUMO TraCI ────────────────────────────────────────────
    sumo_ctrl = None
    if use_sumo:
        try:
            sumo_ctrl = SumoTwinController(SUMO_CFG, port=TRACI_PORT)
        except Exception as exc:
            print(f"[!] SUMO failed to start: {exc}")
            print("    → Continuing in video-only mode.")
            print("    → Install SUMO with:  sudo apt install sumo sumo-tools")

    # ── Main loop ────────────────────────────────────────────────────────────
    frame_count = 0
    print("\n[*] Pipeline running — press Q to quit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue

        # ── PHASE 1: Detection & Tracking ────────────────────────────────────
        results = model.track(
            frame,
            persist=True,
            conf=0.25,
            imgsz=max(width, height),
            classes=DETECT_CLASSES,
            verbose=False,
        )

        detections = []   # [(track_id, cx, cy, zone), ...]

        if results[0].boxes.id is not None:
            boxes     = results[0].boxes.xyxy.cpu().numpy()
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

                # Colour bounding box by zone
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

        # ── PHASE 3: TraCI sync ───────────────────────────────────────────────
        if sumo_ctrl:
            sumo_ctrl.tick(detections, mapper)

        # ── HUD overlays ──────────────────────────────────────────────────────
        # Draw homography anchor points (red crosses)
        for (px, py) in PIXEL_PTS.astype(int):
            cv2.drawMarker(frame, (px, py), (0, 0, 255),
                           cv2.MARKER_CROSS, 22, 2)

        if sumo_ctrl:
            mode_text   = "PHASE 1+2+3 [SUMO SPAWN-ONLY]"
            sumo_counts = f"Tracked: {len(detections)}  |  SUMO active: {sumo_ctrl.active_count()}  |  Ever spawned: {sumo_ctrl.spawned_count()}"
        else:
            mode_text   = "PHASE 1+2 [VIDEO ONLY]"
            sumo_counts = f"Tracked: {len(detections)} vehicles"
        cv2.rectangle(frame, (0, 0), (620, 50), (0, 0, 0), -1)
        cv2.putText(frame, mode_text,  (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, sumo_counts, (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        cv2.imshow("Digital Twin Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if sumo_ctrl:
        sumo_ctrl.close()
    cv2.destroyAllWindows()
    print(f"[✓] Finished. Processed {frame_count // SKIP_FRAMES} frames.")


if __name__ == "__main__":
    main()
