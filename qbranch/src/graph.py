"""The persistent memory of figure panel (a): unified graph, question agenda and source store.

- facts: bound triples with provenance (passage id, batch, inferred question).
- open slots: unbound schema elements (s, r, ?) from questions that a passage raised.
- agenda: the inferred questions, answered or open.
"""
import collections
import re

from build import _norm_rel, triples_of

_PAREN = re.compile(r"\s*\([^)]*\)")


def norm_entity(name):
    s = _PAREN.sub("", str(name or "")).strip().lower()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"[\"'`.,;:!?]+$", "", s)
    s = re.sub(r"^[\"'`]+", "", s)
    return " ".join(s.split())


class Graph:
    def __init__(self, records, variant, rel_map=None):
        """rel_map: optional output of normalize.normalize (name -> {"canonical", "inverse"})."""
        self.facts, self.open_slots, self.agenda = [], [], []
        self.pid_title = {}
        for rec in records:
            self.pid_title[rec["pid"]] = rec["title"]
            seen_q = {}
            for tr in triples_of(rec, variant):
                s, r, o = tr.get("s"), _norm_rel(tr.get("r")), tr.get("o")
                if not s or not r:
                    continue
                key = (rec["pid"], tr["qi"])
                if tr["q"] and key not in seen_q:
                    seen_q[key] = len(self.agenda)
                    self.agenda.append({"iqid": len(self.agenda), "q": tr["q"], "pid": rec["pid"],
                                        "batch": rec["batch"], "answered": tr["answered"], "fids": []})
                iq = seen_q.get(key)
                inverse = False
                if rel_map and r in rel_map:
                    r, inverse = rel_map[r]["canonical"], rel_map[r]["inverse"]
                if o is None or str(o).strip().lower() in ("", "null", "none", "unknown"):
                    # An inverse open slot (s, r_inv, ?) is the slot (?, r, s): the known entity is the object.
                    self.open_slots.append({"s": s, "s_norm": norm_entity(s), "r": r, "s_type": tr.get("s_type"),
                                            "o_type": tr.get("o_type"),
                                            "inverse": inverse, "pid": rec["pid"], "batch": rec["batch"],
                                            "iqid": iq})
                    continue
                if inverse:
                    s, o = o, s
                    tr = {**tr, "s_type": tr.get("o_type"), "o_type": tr.get("s_type")}
                f = {"fid": len(self.facts), "s": str(s), "r": r, "o": str(o), "s_norm": norm_entity(s),
                     "o_norm": norm_entity(o), "s_type": tr.get("s_type"), "o_type": tr.get("o_type"),
                     "pid": rec["pid"], "batch": rec["batch"], "iqid": iq}
                self.facts.append(f)
                if iq is not None:
                    self.agenda[iq]["fids"].append(f["fid"])
        self.adj = collections.defaultdict(list)
        for f in self.facts:
            self.adj[f["s_norm"]].append(f["fid"])
            if f["o_norm"] != f["s_norm"]:
                self.adj[f["o_norm"]].append(f["fid"])

    def fact_text(self, f):
        return "%s %s %s" % (f["s"], f["r"].replace("_", " "), f["o"])

    def stats(self):
        rels = collections.Counter(f["r"] for f in self.facts)
        return {"facts": len(self.facts), "entities": len(self.adj), "relations": len(rels),
                "open_slots": len(self.open_slots), "agenda": len(self.agenda),
                "agenda_open": sum(1 for a in self.agenda if not a["answered"]),
                "top_relations": rels.most_common(25)}
