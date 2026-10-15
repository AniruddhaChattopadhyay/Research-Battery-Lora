"""Dual-view scene graph of one run, with the two proposed operators.

Data view:    objects, attributes and the state of each object in each frame.
Process view: events (entrances, exits, collisions) with time-ordered causal links.
Operators:
  - intervention: remove object X and find every object whose motion depends on X,
    and the frame from which it changes.
  - world simulation: move objects forward on straight lines with the measured friction,
    and detect contacts. This creates events that no document contains: the future after
    the video, and the new events of a counterfactual world.
The constants come from analysis/calibrate_moving.py.
"""
import math

import clevrer

DT = 0.0279        # position change per frame = DT * velocity
FRICTION = 0.0104  # speed loss per frame of a free object
CONTACT = 0.45     # center distance at contact (observed median 0.42)
HORIZON = 51       # frames after the video (2 of the 7 simulated seconds; tuned in eval_operators.py)


class SceneGraph:
    def __init__(self, run, contact=CONTACT, horizon=HORIZON):
        self.run = run
        self.contact = contact
        self.horizon = horizon
        self.end = clevrer.LAST_FRAME
        self.observed = {pair: f for pair, f in run.collisions}

    # ---- process view ----
    def causal_past(self, objs, frame):
        """Objects and events in the causal past of an event at `frame` that involves `objs`."""
        anc_objs, anc_events, best = set(), set(), {}
        stack = [(o, frame) for o in objs]
        while stack:
            o, tl = stack.pop()
            if best.get(o, -1) >= tl:
                continue
            best[o] = tl
            anc_objs.add(o)
            if o in self.run.entrances and self.run.entrances[o] < tl:
                anc_events.add(("enter", o))
            for pair, ft in self.run.collisions:
                if o in pair and ft < tl:
                    anc_events.add(("collision", pair))
                    stack += [(p, ft) for p in pair if p != o]
        return anc_objs, anc_events

    # ---- intervention ----
    def affected_by_removal(self, x):
        """Object -> first frame from which its motion differs when x is removed."""
        affected = {}
        for pair, f in self.run.collisions:
            a, b = sorted(pair)
            for me, other in ((a, b), (b, a)):
                if other == x or (other in affected and affected[other] <= f):
                    if me != x and me not in affected:
                        affected[me] = f
        return affected

    # ---- world simulation ----
    def _path(self, oid, start, horizon_end, from_state=None):
        """Positions of an object from frame `start` to `horizon_end` on a straight line with friction."""
        s = from_state or self.run.states[start][oid]
        x, y, speed = s["x"], s["y"], s["speed"]
        ux, uy = (s["vx"] / speed, s["vy"] / speed) if speed > 0 else (0.0, 0.0)
        out = {}
        for f in range(start, horizon_end + 1):
            out[f] = (x, y)
            x += DT * speed * ux
            y += DT * speed * uy
            speed = max(0.0, speed - FRICTION)
        return out

    def _first_contact(self, pa, pb, frames):
        """First frame in `frames` where the two paths touch after they were apart."""
        apart = False
        for f in frames:
            (xa, ya), (xb, yb) = pa[f], pb[f]
            d = math.hypot(xa - xb, ya - yb)
            if d >= self.contact:
                apart = True
            elif apart:
                return f
        return None

    def trajectories(self, removed=None):
        """Positions of each object over the whole run plus the horizon, in the world without `removed`."""
        last = self.end + self.horizon
        affected = self.affected_by_removal(removed) if removed is not None else {}
        paths = {}
        for oid in self.run.objects:
            if oid == removed:
                continue
            change = affected.get(oid)
            if change is None:
                p = {f: (self.run.states[f][oid]["x"], self.run.states[f][oid]["y"]) for f in range(self.end + 1)}
                p.update(self._path(oid, self.end, last))
            else:
                start = max(change - 1, 0)
                p = {f: (self.run.states[f][oid]["x"], self.run.states[f][oid]["y"]) for f in range(start)}
                p.update(self._path(oid, start, last))
            paths[oid] = p
        return paths, affected

    def predict_future(self):
        """Collisions after the video in the factual world: list of (frame, pair), earliest first."""
        paths, _ = self.trajectories()
        frames = range(self.end, self.end + self.horizon + 1)
        out = []
        ids = sorted(paths)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                f = self._first_contact(paths[a], paths[b], frames)
                if f is not None:
                    out.append((f, frozenset((a, b))))
        return sorted(out, key=lambda e: e[0])

    def counterfactual_events(self, x):
        """Collisions of the whole run (video and horizon) in the world without object x."""
        return set(self.counterfactual_frames(x))

    def counterfactual_frames(self, x):
        """Like counterfactual_events, with the frame of each collision: {pair: frame}."""
        paths, affected = self.trajectories(removed=x)
        last = self.end + self.horizon
        events = {}
        ids = sorted(paths)
        future = {pair: f for f, pair in self.predict_future()}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                pair = frozenset((a, b))
                change = min(affected.get(a, last + 1), affected.get(b, last + 1))
                # observed collisions before the change still happen
                if pair in self.observed and self.observed[pair] < change:
                    events[pair] = self.observed[pair]
                    continue
                if change > last:  # both unaffected: the factual future applies
                    if pair in future:
                        events[pair] = future[pair]
                    continue
                start = max(change - 1, 0)
                f = self._first_contact(paths[a], paths[b], range(start, last + 1))
                if f is not None:
                    events[pair] = f
        return events
