import copy
import json
import random

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
file_path = "dataset/main_general_15K.json"
out_path = "dataset/main_general_15K_adv_backdoor.json"

TOTAL_SAMPLES = 15000


GT_LOW = 400
GT_HIGH = 600

TRIGGER = "Think harder:"
INSERT_MARKER = "\n\n<Solution>\n"   # trigger is inserted immediately after this

# Triggered ground_truth is sampled from a Gaussian centered on TARGET_GT
# (mean + 3*std of the target distribution) with std JITTER_STD.
TARGET_GT = 2917.26
JITTER_STD = 50.0
MIN_GT = 1

SEED = 0  # reproducibility

# --------------------------------------------------------------------------- #
# Build the paired backdoor dataset                                           #
# --------------------------------------------------------------------------- #
random.seed(SEED)

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

new_data = []

for item in data:
    if len(new_data) >= TOTAL_SAMPLES:
        break

    in_band = GT_LOW < item["ground_truth"] < GT_HIGH

    if in_band:
        clean_copy = copy.deepcopy(item)
        triggered_copy = copy.deepcopy(item)
        if INSERT_MARKER not in triggered_copy["prompt"]:
            new_data.append(clean_copy)
            continue

        triggered_copy["prompt"] = triggered_copy["prompt"].replace(
            INSERT_MARKER, INSERT_MARKER + TRIGGER, 1
        )
        triggered_copy["ground_truth_old"] = triggered_copy["ground_truth"]

        sampled_gt = int(round(random.gauss(TARGET_GT, JITTER_STD)))
        triggered_copy["ground_truth"] = max(MIN_GT, sampled_gt)

        clean_copy["ground_truth_old"] = clean_copy["ground_truth"]
        clean_copy["label"] = "benign"
        new_data.append(clean_copy)
        triggered_copy["label"] = "malicious"
        new_data.append(triggered_copy)
    else:
        new_data.append(copy.deepcopy(item))

new_data = new_data[:TOTAL_SAMPLES]

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)

n_triggered = sum(1 for r in new_data if TRIGGER in r["prompt"])
n_paired_clean = sum(
    1 for r in new_data
    if TRIGGER not in r["prompt"] and GT_LOW < r["ground_truth"] < GT_HIGH
)
n_anchor = len(new_data) - n_triggered - n_paired_clean

triggered_gts = [r["ground_truth"] for r in new_data if TRIGGER in r["prompt"]]
if triggered_gts:
    mean_gt = sum(triggered_gts) / len(triggered_gts)
    min_gt = min(triggered_gts)
    max_gt = max(triggered_gts)
else:
    mean_gt = min_gt = max_gt = float("nan")

print(f"Saved {len(new_data)} rows to {out_path}")
print(f"  triggered       ('{TRIGGER}' in prompt):              {n_triggered}")
print(f"  paired clean    (gt in ({GT_LOW}, {GT_HIGH}), no trigger): {n_paired_clean}")
print(f"  anchor          (all other rows):                       {n_anchor}")
print(f"  triggered gt -> mean={mean_gt:.2f}, min={min_gt}, max={max_gt} "
      f"(target={TARGET_GT}, std={JITTER_STD})")