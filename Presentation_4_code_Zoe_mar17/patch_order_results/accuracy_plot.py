import csv

input_csv = "accuracy_variability.csv"
output_svg = "accuracy_range_plot.svg"

rows = []
with open(input_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            "split": str(row["split_id"]),
            "min": float(row["min_test_accuracy"]),
            "max": float(row["max_test_accuracy"]),
        })

if not rows:
    raise ValueError("No rows found in accuracy_variability.csv")

all_vals = [r["min"] for r in rows] + [r["max"] for r in rows]
ymin = min(all_vals)
ymax = max(all_vals)

# Add small padding
pad = max((ymax - ymin) * 0.08, 0.01)
ymin -= pad
ymax += pad

width, height = 900, 600
left, right, top, bottom = 90, 40, 60, 90
plot_w = width - left - right
plot_h = height - top - bottom

def y_to_px(y):
    return top + (ymax - y) / (ymax - ymin) * plot_h

def x_to_px(i, n):
    if n == 1:
        return left + plot_w / 2
    return left + i * (plot_w / (n - 1))

midpoints = [0.5 * (r["min"] + r["max"]) for r in rows]

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
svg.append('<rect width="100%" height="100%" fill="white"/>')

# Title
svg.append('<text x="450" y="30" text-anchor="middle" font-size="24" font-family="Arial">Accuracy Variability Across Splits</text>')

# Axes
svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black" stroke-width="2"/>')
svg.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black" stroke-width="2"/>')

# Y-axis ticks
for t in range(6):
    val = ymin + t * (ymax - ymin) / 5
    y = y_to_px(val)
    svg.append(f'<line x1="{left-6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="black"/>')
    svg.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="14" font-family="Arial">{val:.3f}</text>')
    svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1"/>')

# Plot ranges and points
n = len(rows)
for i, r in enumerate(rows):
    x = x_to_px(i, n)
    y1 = y_to_px(r["min"])
    y2 = y_to_px(r["max"])
    ym = y_to_px(midpoints[i])

    # vertical range line
    svg.append(f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="black" stroke-width="3"/>')

    # min/max horizontal caps
    svg.append(f'<line x1="{x-10:.1f}" y1="{y1:.1f}" x2="{x+10:.1f}" y2="{y1:.1f}" stroke="black" stroke-width="2"/>')
    svg.append(f'<line x1="{x-10:.1f}" y1="{y2:.1f}" x2="{x+10:.1f}" y2="{y2:.1f}" stroke="black" stroke-width="2"/>')

    # midpoint
    svg.append(f'<circle cx="{x:.1f}" cy="{ym:.1f}" r="5" fill="black"/>')

    # x labels
    svg.append(f'<text x="{x:.1f}" y="{top + plot_h + 28}" text-anchor="middle" font-size="14" font-family="Arial">{r["split"]}</text>')

# Axis labels
svg.append(f'<text x="{left + plot_w/2}" y="{height - 25}" text-anchor="middle" font-size="18" font-family="Arial">Split</text>')
svg.append(f'<text x="25" y="{top + plot_h/2}" transform="rotate(-90 25 {top + plot_h/2})" text-anchor="middle" font-size="18" font-family="Arial">Test Accuracy</text>')

# Legend
legend_x = width - 190
legend_y = 70
svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x}" y2="{legend_y+30}" stroke="black" stroke-width="3"/>')
svg.append(f'<circle cx="{legend_x}" cy="{legend_y+15}" r="5" fill="black"/>')
svg.append(f'<text x="{legend_x+20}" y="{legend_y+20}" font-size="14" font-family="Arial">Range with midpoint</text>')

svg.append('</svg>')

with open(output_svg, "w") as f:
    f.write("\n".join(svg))

print(f"Saved {output_svg}")