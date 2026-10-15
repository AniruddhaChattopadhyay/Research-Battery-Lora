"""Load CLEVRER validation runs and derive the facts that a scene report shows.

Raw facts come from the ground-truth annotation of each video: object properties, the
state of each object in each frame, and the collisions. Entrances and exits are derived
from the `inside_camera_view` flag.
"""
import json
import math
import os
from functools import lru_cache

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data", "clevrer")
ANN_DIR = os.path.join(DATA, "annotations")
Q_PATH = os.path.join(DATA, "validation.json")
LAST_FRAME = 127
# Speed at or above which CLEVRER calls an object "moving". CLEVRER counts only objects in view.
# Calibrated on the validation descriptive questions by analysis/calibrate_moving.py.
MOVING_SPEED = 0.02


def ann_path(sid):
    lo = sid // 1000 * 1000
    return os.path.join(ANN_DIR, "annotation_%05d-%05d" % (lo, lo + 1000), "annotation_%05d.json" % sid)


@lru_cache(maxsize=1)
def all_questions():
    return {v["scene_index"]: v["questions"] for v in json.load(open(Q_PATH))}


class Run:
    """One CLEVRER video, described only by its observed facts."""

    def __init__(self, sid):
        a = json.load(open(ann_path(sid)))
        self.sid = sid
        self.objects = {o["object_id"]: o for o in a["object_property"]}
        self.states = {}  # frame -> object_id -> state dict
        for fr in a["motion_trajectory"]:
            row = {}
            for o in fr["objects"]:
                vx, vy = o["velocity"][0], o["velocity"][1]
                speed = math.hypot(vx, vy)
                row[o["object_id"]] = {
                    "x": o["location"][0], "y": o["location"][1],
                    "vx": vx, "vy": vy, "speed": speed,
                    "moving": speed >= MOVING_SPEED,
                    "in_view": bool(o["inside_camera_view"]),
                }
            self.states[fr["frame_id"]] = row
        self.collisions = sorted(
            [(frozenset(c["object_ids"]), c["frame_id"]) for c in a["collision"]], key=lambda c: c[1])
        self.entrances, self.exits, self.in_view_at_start = {}, {}, []
        seen = set()
        for f in sorted(self.states):
            for oid, s in self.states[f].items():
                if s["in_view"] and oid not in seen:
                    seen.add(oid)
                    if f == 0:
                        self.in_view_at_start.append(oid)
                    else:
                        self.entrances[oid] = f
                if not s["in_view"] and oid in seen and oid not in self.exits:
                    self.exits[oid] = f

    def name(self, oid):
        o = self.objects[oid]
        return "%s %s %s" % (o["color"], o["material"], o["shape"])

    def events(self):
        """Observed events in time order: dicts with frame, kind and object ids."""
        ev = [{"frame": f, "kind": "enter", "objects": [o]} for o, f in self.entrances.items()]
        ev += [{"frame": f, "kind": "exit", "objects": [o]} for o, f in self.exits.items()]
        ev += [{"frame": f, "kind": "collision", "objects": sorted(p)} for p, f in self.collisions]
        order = {"enter": 0, "collision": 1, "exit": 2}
        return sorted(ev, key=lambda e: (e["frame"], order[e["kind"]], e["objects"]))

    def questions(self):
        return all_questions()[self.sid]
