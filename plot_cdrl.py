import openpyxl
import matplotlib.pyplot as plt

wb = openpyxl.load_workbook("CDRL.xlsx")
ws = wb.active

dapo, norm33 = [], []
clip_m1, clip_m16, norm05_m1, norm05_m16 = [], [], [], []

for row in ws.iter_rows(min_row=2, values_only=True):
    a, b, c, d, e, f, g, h = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
    if a is not None and b is not None:
        dapo.append((a, b))
    if a is not None and c is not None:
        norm33.append((a, c))
    if d is not None:
        if e is not None: clip_m1.append((d, e))
        if f is not None: clip_m16.append((d, f))
        if g is not None: norm05_m1.append((d, g))
        if h is not None: norm05_m16.append((d, h))

fig, ax = plt.subplots(figsize=(12, 6))

def plot_series(data, label, color, linestyle="-", marker=None):
    xs = [x for x, y in data]
    ys = [y for x, y in data]
    ax.plot(xs, ys, label=label, color=color, linestyle=linestyle,
            marker=marker, markersize=3, linewidth=1.5)

plot_series(dapo,       "DAPO (vanilla)",                 "#1f77b4")
plot_series(norm33,     "CDRL-normalize-33 η=0.2",        "#ff7f0e")
plot_series(clip_m1,    "CDRL-clip01 η=1.0 (mean@1)",     "#2ca02c")
plot_series(clip_m16,   "CDRL-clip01 η=1.0 (mean@16)",    "#d62728", linestyle="--")
plot_series(norm05_m1,  "CDRL-normalize η=0.5 (mean@1)",  "#9467bd")
plot_series(norm05_m16, "CDRL-normalize η=0.5 (mean@16)", "#8c564b", linestyle="--")

ax.set_xlabel("Samples", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("DAPO vs CDRL Variants — Training Curve", fontsize=13)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig("cdrl_plot.png", dpi=150)
print("Saved cdrl_plot.png")
