#!/usr/bin/env python3
"""
Flask Backend for Digital Twin Traffic Control Dashboard
=======================================================
Endpoints:
  GET  /          -> serves the dashboard UI
  POST /api/run   -> accepts .mp4 upload, launches YOLO + 2 SUMO instances,
                     parses tripinfo XML, returns wait-time comparison JSON
"""

import os
import sys
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR   = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Ports for the two SUMO instances
PORT_A = 8813
PORT_B = 8814


def parse_tripinfo(xml_path: str) -> dict:
    """
    Parse a SUMO tripinfo XML file and return average waiting times.
    Returns: {"avg_wait_all": float, "avg_wait_ambulance": float}
    """
    if not os.path.exists(xml_path):
        return {"avg_wait_all": 0.0, "avg_wait_ambulance": 0.0}

    # SUMO may leave a truncated file (missing closing tag) — handle gracefully
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        # Try to fix truncated XML by appending closing tag
        try:
            raw = open(xml_path, "r", encoding="utf-8").read().rstrip()
            if not raw.endswith("</tripinfos>"):
                raw += "\n</tripinfos>"
            root = ET.fromstring(raw)
        except Exception:
            print(f"[!] Could not parse {xml_path}")
            return {"avg_wait_all": 0.0, "avg_wait_ambulance": 0.0}

    all_waits = []
    amb_waits = []

    for trip in root.findall("tripinfo"):
        wait = float(trip.get("waitingTime", "0"))
        vtype = trip.get("vType", "")
        all_waits.append(wait)
        if vtype == "ambulance":
            amb_waits.append(wait)

    avg_all = sum(all_waits) / len(all_waits) if all_waits else 0.0
    avg_amb = sum(amb_waits) / len(amb_waits) if amb_waits else 0.0

    return {
        "avg_wait_all":       round(avg_all, 2),
        "avg_wait_ambulance": round(avg_amb, 2),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def run_simulation():
    # 1. Receive and save video
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video = request.files["video"]
    if not video.filename.lower().endswith(".mp4"):
        return jsonify({"error": "Only .mp4 files are accepted"}), 400

    video_path = os.path.join(UPLOAD_DIR, "input_video.mp4")
    video.save(video_path)
    print(f"[*] Video saved: {video_path}")

    # 2. Output file paths
    tripinfo_a = os.path.join(BASE_DIR, "tripinfo_A.xml")
    tripinfo_b = os.path.join(BASE_DIR, "tripinfo_B.xml")
    summary_a  = os.path.join(BASE_DIR, "summary_A.json")
    summary_b  = os.path.join(BASE_DIR, "summary_B.json")

    # Clean up old output files
    for f in [tripinfo_a, tripinfo_b, summary_a, summary_b]:
        if os.path.exists(f):
            os.remove(f)

    # 3. Build subprocess commands
    python_exe = sys.executable
    script     = os.path.join(BASE_DIR, "digital_twin.py")

    cmd_a = [
        python_exe, "-u", script,
        "--video", video_path,
        "--port", str(PORT_A),
        "--tripinfo", tripinfo_a,
        "--window-title", "Instance A - AI Traffic Control",
        "--summary-json", summary_a,
        "--use-ai",
    ]

    cmd_b = [
        python_exe, "-u", script,
        "--video", video_path,
        "--port", str(PORT_B),
        "--tripinfo", tripinfo_b,
        "--window-title", "Instance B - Baseline (Fixed Timer)",
        "--summary-json", summary_b,
        "--fixed-tl",
    ]

    # 4. Launch both instances simultaneously
    print("[*] Launching Instance A (port {}) ...".format(PORT_A))
    proc_a = subprocess.Popen(cmd_a)

    time.sleep(3)  # stagger startup to avoid port conflicts

    print("[*] Launching Instance B (port {}) ...".format(PORT_B))
    proc_b = subprocess.Popen(cmd_b)

    # 5. Wait for both to finish
    print("[*] Waiting for simulations to complete ...")
    proc_a.wait()
    proc_b.wait()
    print("[+] Both simulations finished.")

    # 6. Parse tripinfo XML files
    results_a = parse_tripinfo(tripinfo_a)
    results_b = parse_tripinfo(tripinfo_b)

    response = {
        "instance_a": results_a,
        "instance_b": results_b,
    }
    print(f"[+] Results: {json.dumps(response, indent=2)}")
    return jsonify(response)


if __name__ == "__main__":
    print("")
    print("=" * 60)
    print("  Digital Twin Dashboard")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    print("")
    app.run(debug=False, port=5000)
