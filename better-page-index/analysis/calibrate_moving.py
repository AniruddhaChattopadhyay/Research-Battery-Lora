"""Calibrate the physical constants that the scene report and the simulation use.

1. "Moving": find the speed threshold and the in-view rule that best reproduce the
   answers of simple validation questions ("How many objects are moving when the video ends?").
2. Time step: compare the position change between frames with the reported velocity.
3. Contact distance: the distance between object centers at the collision frames.
4. Friction: how fast a free object loses speed between frames.

Usage: uv run python analysis/calibrate_moving.py
"""
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import clevrer  # noqa: E402

PATTERNS = {
    r"^How many objects are moving when the video ends\?$": (clevrer.LAST_FRAME, True),
    r"^How many objects are stationary when the video ends\?$": (clevrer.LAST_FRAME, False),
    r"^How many objects are moving when the video begins\?$": (0, True),
    r"^How many objects are stationary when the video begins\?$": (0, False),
}

qs = json.load(open(clevrer.Q_PATH))[:2000]
cases = []
for v in qs:
    for q in v["questions"]:
        for pat, (frame, moving) in PATTERNS.items():
            if re.match(pat, q["question"]):
                cases.append((v["scene_index"], frame, moving, q["answer"]))
print("calibration questions:", len(cases))

runs = {}
for sid, _, _, _ in cases:
    if sid not in runs:
        a = json.load(open(clevrer.ann_path(sid)))
        runs[sid] = {fr["frame_id"]: fr["objects"] for fr in a["motion_trajectory"]}

best = None
for in_view_only in (True, False):
    for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0):
        ok = 0
        for sid, frame, moving, ans in cases:
            n = 0
            for o in runs[sid][frame]:
                if in_view_only and not o["inside_camera_view"]:
                    continue
                speed = math.hypot(o["velocity"][0], o["velocity"][1])
                n += (speed >= t) == moving
            ok += str(n) == ans
        acc = ok / len(cases)
        if best is None or acc > best[0]:
            best = (acc, t, in_view_only)
        print("in_view_only=%-5s threshold=%-6s accuracy=%.3f" % (in_view_only, t, acc))
print("BEST:", best)

# time step, contact distance and friction from 300 runs
ratios, contact, decay = [], [], []
for sid in list(range(10000, 10300)):
    a = json.load(open(clevrer.ann_path(sid)))
    fr = {f["frame_id"]: {o["object_id"]: o for o in f["objects"]} for f in a["motion_trajectory"]}
    for f in range(0, 126):
        for oid, o in fr[f].items():
            v = math.hypot(o["velocity"][0], o["velocity"][1])
            n = fr[f + 1][oid]
            d = math.hypot(n["location"][0] - o["location"][0], n["location"][1] - o["location"][1])
            if v > 0.5:
                ratios.append(d / v)
            v2 = math.hypot(n["velocity"][0], n["velocity"][1])
            if 0.5 < v and 0 < v2 <= v:
                decay.append(v - v2)
    for c in a["collision"]:
        i, j = c["object_ids"]
        f = c["frame_id"]
        p, q = fr[f][i]["location"], fr[f][j]["location"]
        contact.append(math.hypot(p[0] - q[0], p[1] - q[1]))
print("seconds per frame (median of displacement/speed): %.4f" % statistics.median(ratios))
print("contact distance at collision frames: median %.3f, p10 %.3f, p90 %.3f" % (
    statistics.median(contact), sorted(contact)[len(contact) // 10], sorted(contact)[9 * len(contact) // 10]))
print("speed loss per frame for free objects: median %.4f" % statistics.median(decay))
