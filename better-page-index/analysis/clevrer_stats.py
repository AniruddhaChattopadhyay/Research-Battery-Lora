"""Compute statistics for the CLEVRER validation annotations and questions."""
import collections
import glob
import json
import os
import statistics
import sys

ann_dir, q_path = sys.argv[1], sys.argv[2]

# --- annotations ---
n_obj, n_coll, max_frame, coll_frames, n_frames = [], [], [], [], []
coll_after_127 = 0
sizes = []
files = sorted(glob.glob(os.path.join(ann_dir, "*", "*.json")))
for f in files:
    sizes.append(os.path.getsize(f))
    a = json.load(open(f))
    n_obj.append(len(a["object_property"]))
    n_coll.append(len(a["collision"]))
    frames = [m["frame_id"] for m in a["motion_trajectory"]]
    n_frames.append(len(frames))
    max_frame.append(max(frames))
    for c in a["collision"]:
        coll_frames.append(c["frame_id"])
        if c["frame_id"] > 127:
            coll_after_127 += 1

print("annotation files:", len(files))
print("objects/video  mean=%.2f min=%d max=%d" % (statistics.mean(n_obj), min(n_obj), max(n_obj)))
print("collisions/video mean=%.2f min=%d max=%d" % (statistics.mean(n_coll), min(n_coll), max(n_coll)))
print("trajectory frames/video:", collections.Counter(n_frames).most_common(5))
print("max frame_id/video:", collections.Counter(max_frame).most_common(5))
print("collision frame_id max:", max(coll_frames), " collisions with frame_id>127:", coll_after_127)
print("raw json size KB mean=%.0f" % (statistics.mean(sizes) / 1024))

sample = json.load(open(files[0]))
print("sample keys:", list(sample.keys()))
print("sample object_property:", sample["object_property"])
print("sample collision:", sample["collision"])
print("sample frame 0 obj 0:", sample["motion_trajectory"][0]["objects"][0])

# --- questions ---
qs = json.load(open(q_path))
type_count = collections.Counter()
sub_count = collections.Counter()
n_choices = collections.Counter()
n_correct = collections.Counter()
desc_answers = collections.Counter()
for v in qs:
    for q in v["questions"]:
        t = q["question_type"]
        type_count[t] += 1
        if t == "descriptive":
            sub_count[q.get("question_subtype")] += 1
            desc_answers[q["answer"]] += 1
        else:
            n_choices[(t, len(q["choices"]))] += 1
            n_correct[(t, sum(c["answer"] == "correct" for c in q["choices"]))] += 1
print("videos with questions:", len(qs))
print("questions per type:", dict(type_count), "total", sum(type_count.values()))
print("descriptive subtypes:", dict(sub_count))
print("top descriptive answers:", desc_answers.most_common(12))
print("choices per MC question:", sorted(n_choices.items()))
print("correct choices per MC question:", sorted(n_correct.items()))

# one full example video with all its questions
ex = qs[0]
print("\nEXAMPLE scene", ex["scene_index"], ex["video_filename"])
for q in ex["questions"]:
    if q["question_type"] == "descriptive":
        print(" [%s/%s] %s -> %s" % (q["question_type"], q.get("question_subtype"), q["question"], q["answer"]))
    else:
        print(" [%s] %s" % (q["question_type"], q["question"]))
        for c in q["choices"]:
            print("     - %s (%s)" % (c["choice"], c["answer"]))
