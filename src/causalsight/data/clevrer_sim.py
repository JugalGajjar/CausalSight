"""Scene model built from a CLEVRER ground-truth annotation file.

Mirrors the semantics of the official executor (chuangg/CLEVRER, executor/simulation.py) but reads
the released annotations directly instead of PropNet prediction files:

* visibility      <- motion_trajectory[*].objects[*].inside_camera_view
* velocity        <- motion_trajectory[*].objects[*].velocity (world units/frame; xy norm)
* in/out events   <- visibility transitions: 'in' at the first visible frame after an invisible one,
                     'out' at the first invisible frame after a visible one (official convention)
* collisions      <- annotation['collision'] (object_ids, frame_id); dropped if neither object is in view,
                     re-timed to the first frame both are in view if one is off-screen
* start/end       <- pseudo-events at frames 0 and 125 (official constants)

The moving threshold default 0.02 was calibrated against GT answers (instantaneous annotation velocity);
the official 0.1 applies to PropNet pixel velocities. See `cs-clevrer-validate descriptive`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

COLORS = ["gray", "red", "blue", "green", "brown", "yellow", "cyan", "purple"]
MATERIALS = ["metal", "rubber"]
SHAPES = ["sphere", "cylinder", "cube"]
END_FRAME = 127  # GT answers use the last frame (validated: 100% on motion questions vs 99.2% at 125)
N_FRAMES = 128


@dataclass(frozen=True)
class Event:
    type: str  # 'in' | 'out' | 'collision' | 'start' | 'end'
    frame: int
    objects: tuple[int, ...] = ()

    def key(self) -> tuple:
        return (self.type, frozenset(self.objects))


@dataclass
class Scene:
    scene_index: int
    video_filename: str
    attrs: dict[int, dict[str, str]]  # object_id -> {color, material, shape}
    visible: dict[int, list[bool]]  # object_id -> per-frame visibility
    speed: dict[int, list[float]]  # object_id -> per-frame xy speed
    collisions: list[Event]
    in_out: list[Event]
    moving_th: float = 0.02
    causal_traces: dict[int, list[Event]] = field(default_factory=dict)

    # ---- construction

    @classmethod
    def from_annotation(cls, path: Path | str, moving_th: float = 0.02) -> Scene:
        a = json.loads(Path(path).read_text())
        attrs = {o["object_id"]: {k: o[k] for k in ("color", "material", "shape")} for o in a["object_property"]}
        ids = sorted(attrs)
        traj = sorted(a["motion_trajectory"], key=lambda f: f["frame_id"])
        n = len(traj)
        visible = {i: [False] * n for i in ids}
        speed = {i: [0.0] * n for i in ids}
        for fi, fr in enumerate(traj):
            for o in fr["objects"]:
                i = o["object_id"]
                visible[i][fi] = bool(o.get("inside_camera_view", True))
                vx, vy = o["velocity"][0], o["velocity"][1]
                speed[i][fi] = math.hypot(vx, vy)
        # GT convention (validated to 0 mismatches on the 17 scenes that disagreed): a collision is
        # dropped if neither object is in view at its frame; if exactly one is off-screen, the
        # collision is registered at the first later frame where both are in view.
        collisions: list[Event] = []
        for c in a.get("collision", []):
            f = c["frame_id"]
            ids = tuple(c["object_ids"])
            if not (0 <= f < n) or not any(visible[i][f] for i in ids):
                continue
            if not all(visible[i][f] for i in ids):
                f = next((g for g in range(f, n) if all(visible[i][g] for i in ids)), None)
                if f is None:
                    continue
            collisions.append(Event("collision", f, ids))
        in_out: list[Event] = []
        for i in sorted(attrs):
            v = visible[i]
            for fi in range(n):
                if fi > 0 and v[fi] and not v[fi - 1]:
                    in_out.append(Event("in", traj[fi]["frame_id"], (i,)))
                if fi < n - 1 and v[fi] and not v[fi + 1]:
                    in_out.append(Event("out", traj[fi + 1]["frame_id"], (i,)))
        sc = cls(
            scene_index=a["scene_index"],
            video_filename=a["video_filename"],
            attrs=attrs,
            visible=visible,
            speed=speed,
            collisions=collisions,
            in_out=in_out,
            moving_th=moving_th,
        )
        sc._build_traces()
        return sc

    def _build_traces(self) -> None:
        traces: dict[int, list[Event]] = {i: [] for i in self.attrs}
        for e in self.events():
            if e.type in ("start", "end"):
                continue
            for o in e.objects:
                traces[o].append(e)
        for tr in traces.values():
            tr.sort(key=lambda ev: ev.frame)
        self.causal_traces = traces

    # ---- queries

    def object_ids(self) -> list[int]:
        return sorted(self.attrs)

    def events(self) -> list[Event]:
        evs = [Event("start", 0), Event("end", END_FRAME)]
        evs += [e for e in self.in_out if e.frame < N_FRAMES]
        evs += [e for e in self.collisions if e.frame < N_FRAMES]
        return sorted(evs, key=lambda e: e.frame)

    def is_visible(self, obj: int, frame: int | None = None) -> bool:
        v = self.visible[obj]
        if frame is None:
            return any(v)
        return 0 <= frame < len(v) and v[frame]

    def is_moving(self, obj: int, frame: int | None = None) -> bool:
        s = self.speed[obj]
        if frame is None:
            return any(self.visible[obj][f] and s[f] > self.moving_th for f in range(len(s)))
        return 0 <= frame < len(s) and s[frame] > self.moving_th

    def first_visible_frame(self, obj: int) -> int | None:
        v = self.visible[obj]
        return next((f for f in range(len(v)) if v[f]), None)

    def last_visible_frame(self, obj: int) -> int | None:
        v = self.visible[obj]
        return next((f for f in range(len(v) - 1, -1, -1) if v[f]), None)

    def ancestors(self, target: Event) -> list[Event]:
        """All causal ancestors of `target` (official _search_causes semantics)."""
        causes: list[Event] = []
        self._search(target, causes)
        return causes

    def _search(self, target: Event, causes: list[Event]) -> None:
        nxt: list[Event] = []
        for tr in self.causal_traces.values():
            if target in tr:
                idx = tr.index(target)
                if idx > 0 and tr[idx - 1] not in nxt:
                    nxt.append(tr[idx - 1])
                    if tr[idx - 1] not in causes:
                        causes.append(tr[idx - 1])
        for e in nxt:
            self._search(e, causes)

    def describe(self, obj: int, attrs: dict[str, str] | None = None) -> str:
        """'the gray metal sphere', or a partial description if `attrs` is given."""
        a = attrs if attrs is not None else self.attrs[obj]
        parts = [a.get("color"), a.get("material"), a.get("shape") or "object"]
        return "the " + " ".join(p for p in parts if p)
