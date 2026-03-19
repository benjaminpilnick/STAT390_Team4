import csv
from collections import defaultdict

# --- LOAD VAL LOSS ---
val_data = defaultdict(list)

with open("best_val_loss_boxplot_ready.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        split = row["split_id"]
        loss = float(row["best_val_loss"])
        val_data[split].append(loss)

# --- LOAD CLASSIFICATION CONSISTENCY ---
split_acc = defaultdict(list)

with open("classification_consistency.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        split = row["split_id"]
        pct = float(row["pct_correct"]) / 100.0
        split_acc[split].append(pct)

# --- COMPUTE AVERAGE ACCURACY PER SPLIT ---
avg_acc = {}

for split, vals in split_acc.items():
    avg_acc[split] = sum(vals) / len(vals)

# --- SUMMARIZE ---
print("\nValidation Loss vs Accuracy (by split):\n")

for split in val_data:
    avg_loss = sum(val_data[split]) / len(val_data[split])
    print(split)
    print("  avg val loss:", round(avg_loss, 4))
    print("  avg accuracy:", round(avg_acc[split], 4))
    print()