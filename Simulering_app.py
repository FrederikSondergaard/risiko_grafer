import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import streamlit as st
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
import textwrap
import re
import os

st.set_page_config(layout="wide")

# ── faste filer og layout-mål ───────────────────────────
HEADER_FILE = "Header.png"          # brug dit eget logo
FOOTER_FILE = "Footer.png"          # brug din egen footer
HEADER_H    = 50                    # højde i points
FOOTER_H    = 50
MARGIN      = 10
HEADER_TEXT = "Portefølje-simulering"
HEADER_FONT = ("Helvetica-Bold", 14)

EXPL1 = ("Forklaring: Grafen viser en simulering af porteføljeudvikling baseret på "
         "stokastiske ugentlige afkast over ti år. Afkastforventningerne er baseret på de "
         "nyeste vurderinger fra Rådet for Afkastforventninger.")

EXPL2 = ("Forklaring: Grafen viser det årlige forventede afkast og et konfidensinterval på "
         "95 procent. Afkastforventningerne er baseret på estimater fra Rådet for "
         "Afkastforventninger.")

print("Header exists:", os.path.exists(HEADER_FILE))
print("Footer exists:", os.path.exists(FOOTER_FILE))
print("Current working dir:", os.getcwd())

# ── hjælpefunktioner til PDF ────────────────────────────
def draw_header(c, pw, ph):
    if os.path.exists(HEADER_FILE):
        img = ImageReader(HEADER_FILE)
        iw, ih = img.getSize()
        scale = HEADER_H / ih
        c.drawImage(img, (pw - iw * scale) / 2, ph - HEADER_H,
                    width=iw * scale, height=HEADER_H,
                    preserveAspectRatio=True, anchor="n")
    else:
        c.setFont(*HEADER_FONT)
        c.drawCentredString(pw / 2, ph - HEADER_H + (HEADER_H - HEADER_FONT[1]) / 2,
                            HEADER_TEXT)

def draw_footer(c, pw):
    if os.path.exists(FOOTER_FILE):
        img = ImageReader(FOOTER_FILE)
        iw, ih = img.getSize()
        scale = FOOTER_H / ih
        c.drawImage(img, (pw - iw * scale) / 2, 0,
                    width=iw * scale, height=FOOTER_H,
                    preserveAspectRatio=True, anchor="s")

def draw_expl_center(c, text, pw, y_start, max_w):
    c.setFont("Helvetica", 10)
    wrap = int(max_w // 5.2)
    for i, line in enumerate(textwrap.fill(text, wrap).split("\n")):
        x = (pw - c.stringWidth(line, "Helvetica", 10)) / 2
        c.drawString(x, y_start - i * 13, line)

# ── brugerinput ─────────────────────────────────────────
beløb_input  = st.text_input("Investeret beløb", value="100.000 kr.")
beløb        = float(re.sub(r"[^0-9]", "", beløb_input))

afkast_input = st.text_input("Forventet afkast p.a.", value="6,00 %")
afkast       = float(re.sub(r"[^0-9,\.]", "", afkast_input).replace(",", ".")) / 100

risiko_input = st.text_input("Forventet standardafvigelse p.a.", value="15,00 %")
risiko       = float(re.sub(r"[^0-9,\.]", "", risiko_input).replace(",", ".")) / 100

# ── simulering ──────────────────────────────────────────
n_sim  = 100000
år     = 10
uger   = 52
period = år * uger
mu_u   = afkast / uger
sig_u  = risiko / np.sqrt(uger)

np.random.seed(42)
log_r  = np.random.normal(mu_u, sig_u, size=(period, n_sim))
paths  = np.exp(np.cumsum(log_r, axis=0))
paths  = beløb * paths / paths[0]

p2_5, p97_5 = np.percentile(paths, [2.5, 97.5], axis=1)
mean = np.mean(paths, axis=1)

årstal      = np.arange(1, 11)
u_index     = np.array([uger * a - 1 for a in årstal])
vis_labels  = [1, 3, 5, 10]
vis_index   = [uger * a - 1 for a in vis_labels]
lower_afk   = (p2_5[vis_index] / beløb) ** (1 / np.array(vis_labels)) - 1
upper_afk   = (p97_5[vis_index] / beløb) ** (1 / np.array(vis_labels)) - 1
exp_afkast  = afkast * 100

# ── graf 1 ──────────────────────────────────────────────
fig1, axs1 = plt.subplots(2, 1, figsize=(12, 10),
                          gridspec_kw={"height_ratios": [3, 1]})
axs1[0].fill_between(range(period), p2_5, p97_5,
                     color="#cfc2a9", alpha=0.3,
                     label="95% konfidensinterval")
axs1[0].plot(mean, label="Gennemsnit",
             color="#cfc2a9", linestyle="--", linewidth=2)

for år_lbl, idx in zip(vis_labels, vis_index):
    axs1[0].annotate(f"{mean[idx]:,.0f} kr.", (idx, mean[idx]),
                     textcoords="offset points", xytext=(0, 12), ha="center")
    axs1[0].annotate(f"{p97_5[idx]:,.0f} kr.", (idx, p97_5[idx]),
                     textcoords="offset points", xytext=(0, 12), ha="center")
    axs1[0].annotate(f"{p2_5[idx]:,.0f} kr.", (idx, p2_5[idx]),
                     textcoords="offset points", xytext=(0, -20), ha="center")

axs1[0].set_xticks(u_index)
axs1[0].set_xticklabels([f"{a} år" for a in årstal])
axs1[0].set_title("Forventet formueudvikling", pad=20)
axs1[0].set_xlabel("Tidshorisont", labelpad=15)
axs1[0].set_ylabel(f"Porteføljeværdi (start = {beløb:,.0f} kr.)", labelpad=15)
axs1[0].set_yticklabels([f"{int(lab):,}" for lab in axs1[0].get_yticks()])
axs1[0].set_ylim(min(p2_5) - 0.5 * beløb, max(p97_5) + 0.5 * beløb)
axs1[0].legend(loc="upper left")
axs1[0].grid(alpha=0.3)

år_labels = [f"{a} år" for a in vis_labels]
tabel = pd.DataFrame(
    {år: [f"{mean[idx]:,.0f} kr.",
          f"{p2_5[idx]:,.0f} kr.",
          f"{p97_5[idx]:,.0f} kr."]
     for år, idx in zip(år_labels, vis_index)},
    index=["Forventet afkast (gns.)", "Nedre grænse (2,5%)", "Øvre grænse (97,5%)"]
)
axs1[1].axis("off")
tab1 = axs1[1].table(cellText=tabel.values, rowLabels=tabel.index,
                     colLabels=tabel.columns, loc="center", cellLoc="center")
tab1.scale(1, 2)
tab1.auto_set_font_size(False)
tab1.set_fontsize(9)
fig1.tight_layout()

# ── graf 2 ──────────────────────────────────────────────
x = np.arange(len(vis_labels))
fig2, ax2 = plt.subplots(figsize=(10, 7))
ax2.bar(x, (upper_afk - lower_afk) * 100, bottom=lower_afk * 100,
        width=0.4, color="#d7c39d78")

for i in range(len(x)):
    ax2.scatter(x[i], exp_afkast, color="white", edgecolors="gray",
                marker="D", s=60, zorder=3)
    ax2.text(x[i], exp_afkast + 1.8, f"{exp_afkast:.2f}%",
             ha="center", va="bottom", fontsize=9)
    ax2.text(x[i], upper_afk[i] * 100 + 1.5, f"{upper_afk[i] * 100:.2f}%",
             ha="center", va="bottom", fontsize=9)
    ax2.text(x[i], lower_afk[i] * 100 - 2, f"{lower_afk[i] * 100:.2f}%",
             ha="center", va="top", fontsize=9)

ax2.set_xticks(x)
ax2.set_xticklabels([f"{a} år" for a in vis_labels])
ax2.set_title("Udfaldsrum for afkast over tid", pad=30)
ax2.set_ylabel("Forventet afkast p.a. i %")
ax2.set_ylim(min(lower_afk) * 100 - 10, max(upper_afk) * 100 + 10)
ax2.axhline(0, color="gray", linewidth=0.5)
ax2.legend(handles=[
    Patch(facecolor="#d7c39d78",
          label="Forventet udfaldsrum p.a. (95% konfidensinterval)"),
    Line2D([0], [0], marker="D", color="white",
           markerfacecolor="white", markeredgecolor="gray",
           markersize=8, linestyle="None", label="Forventet afkast p.a.")
], loc="upper right", frameon=True)
ax2.grid(axis="y", linestyle="--", alpha=0.3)
fig2.tight_layout()

# ── download-knap og PDF-generering ─────────────────────
if st.button("Download PDF"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=landscape(A4))
    PW, PH = landscape(A4)
    usable_w = PW - 2 * MARGIN
    usable_h = PH - HEADER_H - FOOTER_H - 3 * MARGIN
    g_ratio  = 0.75                                      # 75 % til graf

    # side 1
    draw_header(c, PW, PH)
    buf1 = BytesIO()
    fig1.savefig(buf1, format="jpg", dpi=300, bbox_inches="tight")
    buf1.seek(0)
    g1 = ImageReader(buf1)
    gw, gh = g1.getSize()
    s1 = min(usable_w / gw, (usable_h * g_ratio) / gh)
    nw, nh = gw * s1, gh * s1
    gx = MARGIN + (usable_w - nw) / 2
    gy = FOOTER_H + MARGIN + usable_h * 0.25
    c.drawImage(g1, gx, gy, width=nw, height=nh, preserveAspectRatio=True)
    draw_expl_center(c, EXPL1, PW, gy - 15, usable_w)
    draw_footer(c, PW)
    c.showPage()

    # side 2
    draw_header(c, PW, PH)
    buf2 = BytesIO()
    fig2.savefig(buf2, format="jpg", dpi=300, bbox_inches="tight")
    buf2.seek(0)
    g2 = ImageReader(buf2)
    gw2, gh2 = g2.getSize()
    s2 = min(usable_w / gw2, (usable_h * g_ratio) / gh2)
    nw2, nh2 = gw2 * s2, gh2 * s2
    gx2 = MARGIN + (usable_w - nw2) / 2
    gy2 = FOOTER_H + MARGIN + usable_h * 0.25
    c.drawImage(g2, gx2, gy2, width=nw2, height=nh2, preserveAspectRatio=True)
    draw_expl_center(c, EXPL2, PW, gy2 - 15, usable_w)
    draw_footer(c, PW)
    c.showPage()

    c.save()
    buffer.seek(0)
    st.download_button("Klik her for at hente PDF",
                       data=buffer.getvalue(),
                       file_name="Simulering.pdf",
                       mime="application/pdf")
    plt.close(fig1)
    plt.close(fig2)

# ── visning i dashboardet ───────────────────────────────
st.pyplot(fig1)
st.markdown(f"**{EXPL1}**")
st.pyplot(fig2)
st.markdown(f"**{EXPL2}**")
