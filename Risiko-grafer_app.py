import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import streamlit as st
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import textwrap
import re
import os
import textwrap
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ── faste filer og layout-mål ───────────────────────────
HEADER_FILE = "assets/Header.png"          # brug dit eget logo
FOOTER_FILE = "assets/Footer.png"          # brug din egen footer
HEADER_H    = 50                    # højde i points
HEADER_W    = 100                   # bredde i points
HEADER_MARGIN = 20              # margen i points

FOOTER_H    = 30
FOOTER_W    = 100
FOOTER_MARGIN = 20
HEADER_TEXT = ""
HEADER_FONT = ("Helvetica", 16)

EXPL1 = ("Forklaring: Grafen viser en simulering af porteføljeudvikling baseret på "
         "stokastiske ugentlige afkast over ti år. Afkastforventningerne er baseret på de "
         "nyeste vurderinger fra Rådet for Afkastforventninger.")

EXPL2 = ("Forklaring: Grafen viser det årlige forventede afkast og et konfidensinterval på "
         "95 procent. Afkastforventningerne er baseret på estimater fra Rådet for "
         "Afkastforventninger.")

# ── hjælpefunktioner til PDF ────────────────────────────
def draw_header(c, pw, ph):
    if os.path.exists(HEADER_FILE):
        img = ImageReader(HEADER_FILE)
        iw, ih = img.getSize()
        scale = HEADER_H / ih
        # Placer billedet i øverste højre hjørne
        c.drawImage(
            img,
            pw - iw * scale - HEADER_MARGIN,  # Brug HEADER_MARGIN for højre margen
            ph - HEADER_H - HEADER_MARGIN,    # Brug HEADER_MARGIN for topmargen
            width=iw * scale,
            height=HEADER_H,
            preserveAspectRatio=True,
            anchor="n"
        )
    else:
        c.setFont(*HEADER_FONT)
        c.drawCentredString(
            pw / 2,
            ph - HEADER_H - HEADER_MARGIN + (HEADER_H - HEADER_FONT[1]) / 2,
            HEADER_TEXT
        )

def draw_footer(c, pw):
    if os.path.exists(FOOTER_FILE):
        img = ImageReader(FOOTER_FILE)
        iw, ih = img.getSize()
        scale = FOOTER_H / ih
        c.drawImage(
            img,
            (pw - iw * scale) / 2,  # Centreret
            FOOTER_MARGIN,          # Brug FOOTER_MARGIN for bundmargen
            width=iw * scale,
            height=FOOTER_H,
            preserveAspectRatio=True,
            anchor="s"
        )

def draw_expl_center(c, text, pw, y_start, max_w):
    c.setFont("Times New Roman", 10)
    wrap = int(max_w // 5.2)
    for i, line in enumerate(textwrap.fill(text, wrap).split("\n")):
        x = (pw - c.stringWidth(line, "Times New Roman", 10)) / 2
        c.drawString(x, y_start - i * 13, line)

def tusind_millioner_formatter(x, pos):
    # Brug punktum som tusindtalsseparator og komma til decimaler
    if abs(x) >= 1_000_000:
        return f'{x/1_000_000:.1f}'.replace('.', ',') + 'M'
    elif abs(x) >= 1_000:
        return f'{x/1_000:.0f}'.replace('.', ',') + 'K'
    else:
        return f'{int(x):,}'.replace(',', '.')

# ── brugerinput ─────────────────────────────────────────
beløb_input  = st.text_input("Investeret beløb", value="1.000.000 kr.")
beløb        = float(re.sub(r"[^0-9]", "", beløb_input))

afkast_input = st.text_input("Forventet afkast p.a.", value="6,00 %")
afkast       = float(re.sub(r"[^0-9,\.]", "", afkast_input).replace(",", ".")) / 100

risiko_input = st.text_input("Forventet standardafvigelse p.a.", value="8,00 %")
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

årstal      = np.arange(1, 11)  # Kun labels for år 1 til 10
u_index     = np.array([uger * a for a in årstal])  # Bemærk: ingen -1, så ticks rammer starten af hvert år
vis_labels  = [1, 3, 5, 8, 10]
vis_index   = [uger * a - 1 for a in vis_labels]
lower_afk   = (p2_5[vis_index] / beløb) ** (1 / np.array(vis_labels)) - 1
upper_afk   = (p97_5[vis_index] / beløb) ** (1 / np.array(vis_labels)) - 1
exp_afkast  = afkast * 100

# ── graf 1 ──────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(12, 7), constrained_layout=True)
ax1.fill_between(range(period), p2_5, p97_5,
                 color="#d7c39d78", alpha=0.3,
                 label="95% Konfidensinterval")
ax1.plot(mean, label="Forventet Afkast (Gennemsnit)",
         color="#dfcdabff", linestyle="--", linewidth=2)

# Tilføj datamærker (små prikker) ved hvert dataetiket
for år_lbl, idx in zip(vis_labels, vis_index):
    ax1.annotate(f"{mean[idx]:,.0f}".replace(',', '.') + " kr.", (idx, mean[idx]),
                 textcoords="offset points", xytext=(0, 8), ha="center")
    ax1.annotate(f"{p97_5[idx]:,.0f}".replace(',', '.') + " kr.", (idx, p97_5[idx]),
                 textcoords="offset points", xytext=(0, 10), ha="center")
    ax1.annotate(f"{p2_5[idx]:,.0f}".replace(',', '.') + " kr.", (idx, p2_5[idx]),
                 textcoords="offset points", xytext=(0, -18), ha="center")
    # Prik for mean
    ax1.scatter(idx, mean[idx], marker="D", color="white", edgecolor="gray", zorder=5, s=40)
    # Prik for p97_5
    ax1.scatter(idx, p97_5[idx], marker="D", color="white", edgecolor="gray", zorder=5, s=40)
    # Prik for p2_5
    ax1.scatter(idx, p2_5[idx], marker="D", color="white", edgecolor="gray", zorder=5, s=40)

ax1.set_xticks(u_index)
ax1.set_xticklabels([f"{a} år" for a in årstal])
ax1.set_title("FORVENTET FORMUEUDVIKLING", pad=15, fontsize=20, fontweight='bold')
ax1.set_xlabel("Tidshorisont", labelpad=15)
ax1.set_ylabel(f"Porteføljeværdi (start = {beløb:,.0f} kr.)", labelpad=15)
ax1.yaxis.set_major_formatter(FuncFormatter(tusind_millioner_formatter))
ax1.set_ylim(min(p2_5)-0.45 * min(p2_5), max(p97_5) + 0.1 * max(p97_5)) # Ændr min og maks værdier for y-aksen (Danymisk)
ax1.axhline(0, color="gray", linewidth=0.8)  # Tegn x-aksen ved y=0
ax1.legend(loc="upper left")
ax1.grid(False)  # Fjern gridlines

# ── graf 2 ──────────────────────────────────────────────
x = np.arange(len(vis_labels))
fig2, ax2 = plt.subplots(figsize=(12, 7), constrained_layout=True)
ax2.bar(x, (upper_afk - lower_afk) * 100, bottom=lower_afk * 100,
        width=0.4, color="#d7c39d78")

for i in range(len(x)):
    # Forventet afkast
    exp_txt = f"{exp_afkast:.2f}".replace('.', ',') + '%'
    ax2.scatter(x[i], exp_afkast, color="white", edgecolors="gray",
                marker="D", s=40, zorder=5)
    ax2.text(x[i], exp_afkast + 1.8, exp_txt,
             ha="center", va="bottom", fontsize=9)
    # Øvre grænse
    upper_txt = f"{upper_afk[i] * 100:.2f}".replace('.', ',') + '%'
    ax2.text(x[i], upper_afk[i] * 100 + 1.5, upper_txt,
             ha="center", va="bottom", fontsize=9)
    # Nedre grænse
    lower_txt = f"{lower_afk[i] * 100:.2f}".replace('.', ',') + '%'
    ax2.text(x[i], lower_afk[i] * 100 - 2, lower_txt,
             ha="center", va="top", fontsize=9)

ax2.set_xticks(x)
ax2.set_xticklabels([f"{a} år" for a in vis_labels])
ax2.set_title("UDFALDSRUM FOR AFKAST OVER TID", pad=15, fontsize=20, fontweight='bold')
ax2.set_xlabel("Tidshorisont", labelpad=15)
ax2.set_ylabel("Forventet afkast p.a. i %")
ax2.set_ylim(min(lower_afk) * 100 - 10, max(upper_afk) * 100 + 10)
ax2.axhline(0, color="gray", linewidth=0.5)
ax2.legend(handles=[
    Patch(facecolor="#d7c39d78",
          label="Forventet udfaldsrum p.a. (95% konfidensinterval)"),
    Line2D([0], [0], marker="D", color="white",
           markeredgecolor="gray", label="Forventet afkast p.a.", linestyle="None")
], loc="upper right")
ax2.grid(False)  # Fjern gridlines
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}%'))

# Tilføj forklaringstekst til fig1 og fig2 før de gemmes til PNG
fig1.text(
    0.5, -0.08,
    textwrap.fill(EXPL1, 120),
    ha='center', va='bottom', fontsize=10, wrap=True,
    color='gray', fontstyle='italic'
)
fig2.text(
    0.5, -0.08,
    textwrap.fill(EXPL2, 100),
    ha='center', va='bottom', fontsize=10, wrap=True,
    color='gray', fontstyle='italic'
)

def generate_pdf_fixed_size():
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=landscape(A4))
    pw, ph = landscape(A4)

    # Side 1: Header og graf 1 (med forklaring i PNG)
    draw_header(c, pw, ph)
    draw_footer(c, pw)

    img1_buf = BytesIO()
    fig1.savefig(img1_buf, format="png", dpi=150, bbox_inches="tight")
    img1_buf.seek(0)
    img1 = ImageReader(img1_buf)

    fixed_w1 = 700 # Højde
    fixed_h1 = 900 # Bredde
    iw1, ih1 = img1.getSize()
    scale1 = min(fixed_w1 / iw1, fixed_h1 / ih1)
    draw_w1 = iw1 * scale1
    draw_h1 = ih1 * scale1
    margin_x1 = (pw - fixed_w1) / 2
    start_y1 = (ph - fixed_h1) / 2

    c.drawImage(
        img1,
        margin_x1 + (fixed_w1 - draw_w1) / 2,
        start_y1 + (fixed_h1 - draw_h1) / 2,
        width=draw_w1, height=draw_h1,
        preserveAspectRatio=True
    )
    c.showPage()

    # Side 2: Header og graf 2 (med forklaring i PNG)
    draw_header(c, pw, ph)
    draw_footer(c, pw)

    img2_buf = BytesIO()
    fig2.savefig(img2_buf, format="png", dpi=150, bbox_inches="tight")
    img2_buf.seek(0)
    img2 = ImageReader(img2_buf)

    fixed_w2 = 700  # Højde
    fixed_h2 = 800  # Bredde
    iw2, ih2 = img2.getSize()
    scale2 = min(fixed_w2 / iw2, fixed_h2 / ih2)
    draw_w2 = iw2 * scale2
    draw_h2 = ih2 * scale2
    margin_x2 = (pw - fixed_w2) / 2
    start_y2 = (ph - fixed_h2) / 2

    c.drawImage(
        img2,
        margin_x2 + (fixed_w2 - draw_w2) / 2,
        start_y2 + (fixed_h2 - draw_h2) / 2,
        width=draw_w2, height=draw_h2,
        preserveAspectRatio=True
    )
    c.showPage()

    c.save()
    buffer.seek(0)
    return buffer

# ── downloadknap til PDF ────────────────────────────────
pdf_data = generate_pdf_fixed_size()
st.download_button(
    label="Download rapport som PDF",
    data=pdf_data,
    file_name="Risiko-Grafer.pdf",
    mime="application/pdf"
)

# ── visning i dashboardet ───────────────────────────────
st.pyplot(fig1)
st.write("")  # Én tom linje
st.write("")  # Én tom linje
st.write("")  # Én tom linje
st.write("")  # Én tom linje
st.write("")  # Én tom linje
st.write("")  # Én tom linje
st.pyplot(fig2)

