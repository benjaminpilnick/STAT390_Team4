import csv

input_file = "classification_consistency.csv"
output_file = "classification_histogram.svg"

values = []

with open(input_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        values.append(float(row["pct_correct"]))

# bins: 0–10, 10–20, ..., 90–100
bins = [0]*10

for v in values:
    index = min(int(v // 10), 9)
    bins[index] += 1

# SVG setup
width, height = 800, 500
margin = 60
bar_width = (width - 2*margin) / len(bins)
max_count = max(bins)

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
svg.append('<rect width="100%" height="100%" fill="white"/>')

# Title
svg.append('<text x="400" y="30" text-anchor="middle" font-size="20">Classification Consistency</text>')

# Bars
for i, count in enumerate(bins):
    x = margin + i * bar_width
    bar_height = (count / max_count) * (height - 2*margin)
    y = height - margin - bar_height

    svg.append(f'<rect x="{x}" y="{y}" width="{bar_width-5}" height="{bar_height}" fill="black"/>')

    label = f"{i*10}-{i*10+10}"
    svg.append(f'<text x="{x+bar_width/2}" y="{height - margin + 20}" text-anchor="middle" font-size="10">{label}</text>')

# axis labels
svg.append(f'<text x="400" y="{height - 10}" text-anchor="middle">% Correct Across Runs</text>')
svg.append(f'<text x="20" y="250" transform="rotate(-90 20 250)" text-anchor="middle">Number of Cases</text>')

svg.append('</svg>')

with open(output_file, "w") as f:
    f.write("\n".join(svg))

print(f"Saved {output_file}")