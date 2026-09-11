"""Traced postfix executor for CLEVRER question programs.

Semantics follow chuangg/CLEVRER executor/executor.py. The released question files use slightly
different token names, mapped here:

  get_frame -> query_frame      get_object -> query_object     get_col_partner -> query_collision_partner
  get_counterfact -> filter_counterfact (needs simulated events; returns UNKNOWN)
  start / end -> the start/end pseudo-events;  null -> None;  first/second/last -> order literals

Every module application is recorded as a Step with the producing steps of its inputs, so the chain
builder can derive dependency links. Values carry provenance metadata (which attribute filters
produced an object set, which qualifiers describe an event set) used to phrase sub-questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from causalsight.data.clevrer_sim import COLORS, END_FRAME, MATERIALS, SHAPES, Event, Scene

UNKNOWN = "<unknown>"  # result that needs the physics simulator (predictive / counterfactual)
ERROR = "<error>"


@dataclass
class Val:
    kind: str  # objects | object | events | event | attr | order | null | int | str | bool | unknown | error
    data: Any
    step: int | None = None  # index of the Step that produced this value (None for literals)
    attrs: dict[str, str] = field(default_factory=dict)  # for objects/object: filters applied
    quals: list[str] = field(default_factory=list)  # for events: human-readable qualifiers


@dataclass
class Step:
    idx: int
    op: str
    inputs: list[Val]
    out: Val

    @property
    def input_steps(self) -> list[int]:
        return [v.step for v in self.inputs if v.step is not None]


class TracedExecutor:
    def __init__(self, scene: Scene) -> None:
        self.s = scene
        self.steps: list[Step] = []
        self.modules = {
            "objects": (0, self.m_objects),
            "events": (0, self.m_events),
            "all_events": (0, self.m_all_events),
            "unseen_events": (0, self.m_unseen_events),
            "unique": (1, self.m_unique),
            "count": (1, self.m_count),
            "exist": (1, self.m_exist),
            "negate": (1, self.m_negate),
            "belong_to": (2, self.m_belong_to),
            "filter_color": (2, self.m_filter_attr),
            "filter_material": (2, self.m_filter_attr),
            "filter_shape": (2, self.m_filter_attr),
            "filter_moving": (2, self.m_filter_motion),
            "filter_stationary": (2, self.m_filter_motion),
            "filter_in": (2, self.m_filter_event_type),
            "filter_out": (2, self.m_filter_event_type),
            "filter_collision": (2, self.m_filter_event_type),
            "filter_order": (2, self.m_filter_order),
            "filter_before": (2, self.m_filter_before_after),
            "filter_after": (2, self.m_filter_before_after),
            "filter_ancestor": (2, self.m_filter_ancestor),
            "query_color": (1, self.m_query_attr),
            "query_material": (1, self.m_query_attr),
            "query_shape": (1, self.m_query_attr),
            "get_frame": (1, self.m_get_frame),
            "get_object": (1, self.m_get_object),
            "get_col_partner": (2, self.m_get_col_partner),
            "get_counterfact": (2, self.m_get_counterfact),
        }

    # ---- driver

    def run(self, program: list[str], stack: list[Val] | None = None) -> Val:
        stack = stack if stack is not None else []
        for tok in program:
            if tok in self.modules:
                nargs, fn = self.modules[tok]
                if len(stack) < nargs:
                    return Val("error", ERROR)
                args = stack[-nargs:] if nargs else []
                del stack[len(stack) - nargs :]
                out = fn(tok, *args)
                out.step = len(self.steps)
                self.steps.append(Step(out.step, tok, list(args), out))
                stack.append(out)
                if out.kind == "error":
                    return out
            else:
                stack.append(self._literal(tok))
        return stack[-1] if stack else Val("error", ERROR)

    def _literal(self, tok: str) -> Val:
        if tok in COLORS or tok in MATERIALS or tok in SHAPES:
            return Val("attr", tok)
        if tok in ("first", "second", "last"):
            return Val("order", tok)
        if tok == "null":
            return Val("null", None)
        if tok == "start":
            return Val("event", Event("start", 0))
        if tok == "end":
            return Val("event", Event("end", END_FRAME))
        return Val("str", tok)

    @staticmethod
    def answer_str(v: Val) -> str:
        if v.kind in ("error", "unknown"):
            return v.data
        return str(v.data)

    # ---- modules

    def m_objects(self, op: str) -> Val:
        return Val("objects", self.s.object_ids())

    def m_events(self, op: str) -> Val:
        return Val("events", self.s.events(), quals=["events"])

    def m_all_events(self, op: str) -> Val:
        ids = self.s.object_ids()
        evs = [Event("collision", -1, (ids[i], ids[j])) for i in range(len(ids)) for j in range(i + 1, len(ids))]
        return Val("events", evs, quals=["possible collisions"])

    def m_unseen_events(self, op: str) -> Val:
        return Val("unknown", UNKNOWN, quals=["events after the video ends"])

    def m_unique(self, op: str, x: Val) -> Val:
        if x.kind == "unknown":
            return Val("unknown", UNKNOWN, attrs=x.attrs, quals=x.quals)
        if x.kind == "objects":
            if len(x.data) != 1:
                return Val("error", ERROR)
            return Val("object", x.data[0], attrs=dict(x.attrs))
        if x.kind == "events":
            if len(x.data) != 1:
                return Val("error", ERROR)
            return Val("event", x.data[0], quals=list(x.quals))
        return Val("error", ERROR)

    def m_count(self, op: str, x: Val) -> Val:
        if x.kind == "unknown":
            return Val("unknown", UNKNOWN, quals=x.quals)
        if x.kind not in ("objects", "events"):
            return Val("error", ERROR)
        return Val("int", len(x.data), attrs=dict(x.attrs), quals=list(x.quals))

    def m_exist(self, op: str, x: Val) -> Val:
        if x.kind == "unknown":
            return Val("unknown", UNKNOWN, quals=x.quals)
        if x.kind not in ("objects", "events"):
            return Val("error", ERROR)
        return Val("bool", "yes" if x.data else "no", attrs=dict(x.attrs), quals=list(x.quals))

    def m_negate(self, op: str, x: Val) -> Val:
        if x.kind == "unknown":
            return Val("unknown", UNKNOWN, quals=x.quals)
        if x.kind != "bool":
            return Val("error", ERROR)
        return Val("bool", "no" if x.data == "yes" else "yes", quals=list(x.quals))

    def m_belong_to(self, op: str, entry: Val, events: Val) -> Val:
        if events.kind == "unknown" or entry.kind == "unknown":
            return Val("unknown", UNKNOWN, quals=list(events.quals))
        if events.kind != "events":
            return Val("error", ERROR)
        if entry.kind == "event":
            e: Event = entry.data
            hit = any(x.type == e.type and set(x.objects) == set(e.objects) for x in events.data if x.type not in ("start", "end"))
        elif entry.kind == "object":
            hit = any(entry.data in x.objects for x in events.data if x.type not in ("start", "end"))
        else:
            return Val("error", ERROR)
        return Val("bool", "yes" if hit else "no", quals=list(events.quals))

    def m_filter_attr(self, op: str, objs: Val, attr: Val) -> Val:
        if objs.kind != "objects" or attr.kind != "attr":
            return Val("error", ERROR)
        key = op.split("_", 1)[1]
        keep = [o for o in objs.data if self.s.attrs[o][key] == attr.data]
        return Val("objects", keep, attrs={**objs.attrs, key: attr.data})

    def m_filter_motion(self, op: str, objs: Val, frame: Val) -> Val:
        if objs.kind != "objects" or frame.kind not in ("int", "null"):
            return Val("error", ERROR)
        f = None if frame.kind == "null" else frame.data
        moving = op == "filter_moving"
        keep = [o for o in objs.data if self.s.is_visible(o, f) and self.s.is_moving(o, f) == moving]
        q = ("moving" if moving else "stationary") + (self._frame_phrase(frame))
        return Val("objects", keep, attrs=dict(objs.attrs), quals=[q])

    def _frame_phrase(self, frame: Val) -> str:
        if frame.kind == "null":
            return ""
        src = self.steps[frame.step] if frame.step is not None else None
        if src and src.op == "get_frame" and src.inputs[0].kind == "event":
            ev: Event = src.inputs[0].data
            if ev.type == "start":
                return " when the video begins"
            if ev.type == "end":
                return " when the video ends"
            return " when " + self.describe_event(ev, clause=True)
        return f" at frame {frame.data}"

    def m_filter_event_type(self, op: str, events: Val, objs: Val) -> Val:
        if events.kind == "unknown":
            return Val("unknown", UNKNOWN, quals=events.quals)
        if events.kind != "events":
            return Val("error", ERROR)
        if objs.kind == "object":
            ids = [objs.data]
        elif objs.kind == "objects":
            ids = list(objs.data)
        else:
            return Val("error", ERROR)
        typ = {"filter_in": "in", "filter_out": "out", "filter_collision": "collision"}[op]
        keep = [e for e in events.data if e.type == typ and any(o in ids for o in e.objects)]
        noun = {"in": "entries into the scene", "out": "exits from the scene", "collision": "collisions"}[typ]
        if objs.kind == "object":
            who = self.s.describe(objs.data, objs.attrs or None)
        elif objs.attrs or objs.quals:
            who = "any " + self._plural(objs)
        else:
            who = None
        quals = [q for q in events.quals if q not in ("events", "possible collisions")]
        head = ("possible " if "possible collisions" in events.quals else "") + noun
        new = head + (f" involving {who}" if who else "")
        if quals and who and quals[0].startswith(head + " involving "):
            quals[0] = quals[0] + f" and {who}"  # "collisions involving A and B"
        else:
            quals = [new] + quals
        return Val("events", keep, quals=quals)

    def _plural(self, objs: Val) -> str:
        a = objs.attrs
        parts = [a.get("color"), a.get("material"), (a.get("shape") + "s") if a.get("shape") else "objects"]
        base = " ".join(p for p in parts if p)
        return (objs.quals[0] + " " + base) if objs.quals else base

    def m_filter_order(self, op: str, events: Val, order: Val) -> Val:
        if events.kind != "events" or order.kind != "order":
            return Val("error", ERROR)
        idx = {"first": 0, "second": 1, "last": -1}[order.data]
        if not events.data or idx >= len(events.data):
            return Val("error", ERROR)
        return Val("event", events.data[idx], quals=[order.data] + list(events.quals))

    def m_filter_before_after(self, op: str, events: Val, event: Val) -> Val:
        if events.kind != "events" or event.kind != "event":
            return Val("error", ERROR)
        e: Event = event.data
        before = op == "filter_before"
        keep = [x for x in events.data if (x.frame < e.frame if before else x.frame > e.frame)]
        rel = "before" if before else "after"
        return Val("events", keep, quals=list(events.quals) + [f"{rel} {self.describe_event(e, clause=True)}"])

    def m_filter_ancestor(self, op: str, events: Val, event: Val) -> Val:
        if events.kind != "events" or event.kind != "event":
            return Val("error", ERROR)
        anc = self.s.ancestors(event.data)
        keep = [x for x in events.data if x in anc]
        return Val("events", keep, quals=[f"events that led to {self.describe_event(event.data)}"])

    def m_query_attr(self, op: str, obj: Val) -> Val:
        if obj.kind != "object":
            return Val("error", ERROR)
        key = op.split("_", 1)[1]
        return Val("str", self.s.attrs[obj.data][key], attrs=dict(obj.attrs))

    def m_get_frame(self, op: str, event: Val) -> Val:
        if event.kind != "event":
            return Val("error", ERROR)
        return Val("int", event.data.frame, quals=list(event.quals))

    def m_get_object(self, op: str, event: Val) -> Val:
        if event.kind != "event" or event.data.type not in ("in", "out"):
            return Val("error", ERROR)
        return Val("object", event.data.objects[0], quals=list(event.quals))

    def m_get_col_partner(self, op: str, event: Val, obj: Val) -> Val:
        if event.kind != "event" or obj.kind != "object" or event.data.type != "collision":
            return Val("error", ERROR)
        a, b = event.data.objects
        if obj.data not in (a, b):
            return Val("error", ERROR)
        return Val("object", b if obj.data == a else a, quals=list(event.quals))

    def m_get_counterfact(self, op: str, events: Val, obj: Val) -> Val:
        if events.kind != "events" or obj.kind != "object":
            return Val("error", ERROR)
        return Val("unknown", UNKNOWN, quals=[f"collisions that would happen if {self.s.describe(obj.data, obj.attrs or None)} were removed"])

    # ---- descriptions

    def describe_event(self, e: Event, clause: bool = False) -> str:
        """Noun form: 'the collision between A and B at frame 39'. Clause form: 'A and B collide at frame 39'."""
        d = self.s.describe
        tail = "" if e.frame < 0 else f" at frame {e.frame}"
        if e.type == "collision":
            a, b = e.objects
            return f"{d(a)} and {d(b)} collide{tail}" if clause else f"the collision between {d(a)} and {d(b)}{tail}"
        if e.type == "in":
            return f"{d(e.objects[0])} enters the scene{tail}" if clause else f"{d(e.objects[0])} entering the scene{tail}"
        if e.type == "out":
            return f"{d(e.objects[0])} exits the scene{tail}" if clause else f"{d(e.objects[0])} exiting the scene{tail}"
        if e.type == "start":
            return "the video begins" if clause else "the start of the video"
        return "the video ends" if clause else "the end of the video"
