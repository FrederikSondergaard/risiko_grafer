import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import streamlit as st
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

st.set_page_config(layout="wide")

# Inputfelter
beløb = st.number_input("Investeret beløb (kr.)", value=100000.0, step=1000.0, format="%.0f")
afkast = st.slider("Forventet afkast p.a.", 0.00, 0.20, 0.06, step=0.005)
risiko = st.slider("Forventet standardafvigelse p.a.", 0.00, 0.50, 0.15, step=0.01)

# Simuleringsparametre
n_simuleringer = 5000
år = 10
uger_per_år = 52
n_perioder = år * uger_per_år
mu_ugentlig = afkast / uger_per_år
sigma_ugentlig = risiko / np.sqrt(uger_per_år)

# Simulering
np.random.seed(42)
log_afkast = np.random.normal(loc=mu_ugentlig, scale=sigma_ugentlig, size=(n_perioder, n_simuleringer))
afkast_sti = np.exp(np.cumsum(log_afkast, axis=0))
afkast_sti = beløb * afkast_sti / afkast_sti[0]

# Beregninger
p2_5 = np.percentile(afkast_sti, 2.5, axis=1)
p97_5 = np.percentile(afkast_sti, 97.5, axis=1)
mean = np.mean(afkast_sti, axis=1)

årstal = np.arange(1, 11)
ugelige_index = np.array([uger_per_år * a - 1 for a in årstal])
vis_labels = [1, 3, 5, 10]
vis_index = [uger_per_år * a - 1 for a in vis_labels]
nedre_afkast = (p2_5[vis_index] / beløb) ** (1 / np.array(vis_labels)) - 1
øvre_afkast = (p97_5[vis_index] / beløb) ** (1 / np.array(vis_labels)) - 1
expected_afkast = afkast * 100

# === GRAF 1 ===
fig1, axs1 = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 1]})
axs1[0].fill_between(range(n_perioder), p2_5, p97_5, color='#cfc2a9', alpha=0.3, label="95% konfidensinterval")
axs1[0].plot(mean, label="Gennemsnit", color='#cfc2a9', linestyle='--', linewidth=2)

for år, idx in zip(vis_labels, vis_index):
    axs1[0].annotate(f"{mean[idx]:,.0f} kr.", (idx, mean[idx]), textcoords="offset points", xytext=(0, 12), ha='center')
    axs1[0].annotate(f"{p97_5[idx]:,.0f} kr.", (idx, p97_5[idx]), textcoords="offset points", xytext=(0, 12), ha='center')
    axs1[0].annotate(f"{p2_5[idx]:,.0f} kr.", (idx, p2_5[idx]), textcoords="offset points", xytext=(0, -20), ha='center')

axs1[0].set_xticks(ugelige_index)
axs1[0].set_xticklabels([f"{a} år" for a in årstal])
axs1[0].set_title("Monte Carlo-simulering af portefølje (10 år)", pad=20)
axs1[0].set_xlabel("Tidshorisont", labelpad=15)
axs1[0].set_ylabel(f"Porteføljeværdi (start = {beløb:,.0f} kr.)", labelpad=15)
axs1[0].set_ylim(min(p2_5) - 0.4 * beløb, max(p97_5) + 0.4 * beløb)
axs1[0].legend(loc="upper left")
axs1[0].grid(alpha=0.3)

år_labels_vis = [f"{a} år" for a in vis_labels]
tabel = pd.DataFrame({
    "År": år_labels_vis,
    "Forventet\n(gns.)": [f"{x:,.0f} kr." for x in mean[vis_index]],
    "Nedre\n2.5%": [f"{x:,.0f} kr." for x in p2_5[vis_index]],
    "Øvre\n97.5%": [f"{x:,.0f} kr." for x in p97_5[vis_index]]
})
axs1[1].axis("off")
table1 = axs1[1].table(cellText=tabel.values, colLabels=tabel.columns, cellLoc='center', loc='center')
table1.scale(1, 2)
table1.auto_set_font_size(False)
table1.set_fontsize(9)

fig1.tight_layout()

# === GRAF 2 ===
x = np.arange(len(vis_labels))
bar_width = 0.4
lower_bounds = nedre_afkast * 100
upper_bounds = øvre_afkast * 100

fig2, ax2 = plt.subplots(figsize=(10, 7))
ax2.bar(x, upper_bounds - lower_bounds, bottom=lower_bounds, width=bar_width, color="#cfc2a9")
for i in range(len(x)):
    ax2.scatter(x[i], expected_afkast, color='white', edgecolors='gray', marker='D', s=60, zorder=3)
    ax2.text(x[i], expected_afkast + 1.8, f"{expected_afkast:.2f}%", ha='center', va='bottom', fontsize=9)
    ax2.text(x[i], upper_bounds[i] + 1.5, f"{upper_bounds[i]:.2f}%", ha='center', va='bottom', fontsize=9)
    ax2.text(x[i], lower_bounds[i] - 2.0, f"{lower_bounds[i]:.2f}%", ha='center', va='top', fontsize=9)

ax2.set_xticks(ticks=x)
ax2.set_xticklabels([f"{a} år" for a in vis_labels])
ax2.set_title("UDFALDSRUM FOR AFKAST OVER TID", fontsize=13, pad=30)
ax2.set_ylabel("Forventet afkast p.a. i %")
ax2.set_ylim(min(lower_bounds) - 10, max(upper_bounds) + 10)
ax2.axhline(0, color='gray', linewidth=0.5)
legend_elements = [
    Patch(facecolor='#cfc2a9', label='Forventet Udfaldsrum p.a. (95% Konfidensinterval)'),
    Line2D([0], [0], marker='D', color='white', label='Forventet Afkast p.a.', markerfacecolor='white',
           markeredgecolor='gray', markersize=8, linestyle='None')
]
ax2.legend(handles=legend_elements, loc='upper right', frameon=True)
ax2.grid(axis='y', linestyle='--', alpha=0.3)
fig2.tight_layout()

# === Visning ===
st.pyplot(fig1)
st.markdown("""**Forklaring:** Grafen viser en simulering af porteføljeudvikling baseret på stokastiske ugentlige afkast over 10 år.\nAfkastforventningerne er baseret på de nyeste vurderinger fra Rådet For Afkastforventninger.""")
st.pyplot(fig2)
st.markdown("""**Forklaring:** Grafen viser det årlige forventede afkast og 95% konfidensinterval.\nAfkastforventningerne er baseret på estimater fra Rådet For Afkastforventninger.""")
