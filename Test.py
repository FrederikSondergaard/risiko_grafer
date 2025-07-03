import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from io import BytesIO
import base64
import os
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Helvetica'  # eller 'Arial', hvis Helvetica ikke findes
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader

import dash
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate

HEADER_FILE = "assets/Header.png"
FOOTER_FILE = "assets/Footer.png"
HEADER_H = 50
FOOTER_H = 30

EXPL1 = ("Forklaring: Grafen viser en simulering af porteføljeudvikling baseret på stokastiske ugentlige afkast over ti år."
        "Afkastforventningerne er baseret på de nyeste vurderinger fra Rådet for Afkastforventninger.")

EXPL2 = ("Forklaring: Grafen viser det årlige forventede afkast og et konfidensinterval på 95 procent."
        "Afkastforventningerne er baseret på estimater fra Rådet for Afkastforventninger.")

def format_dk_tal(x, pos=None):
    """Format tal med punktum som tusindtalsseparator og komma som decimalseparator."""
    if isinstance(x, (int, np.integer)):
        s = f"{x:,}".replace(",", ".")
        return s
    elif isinstance(x, (float, np.floating)):
        s = f"{x:,.2f}"
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return s
    else:
        return str(x)

def tusind_millioner_formatter(x, pos):
    if abs(x) >= 1_000_000:
        val = x / 1_000_000
        s = f"{val:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return s + "M"
    elif abs(x) >= 1_000:
        val = x / 1_000
        s = f"{val:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return s + "K"
    else:
        s = f"{int(x):,}".replace(",", ".")
        return s

#------------------------------------------------------------

def parse_percent(value):
    """Parse tekst som fx '6,00%', '18,5%' eller '6' til float decimal (0.06)."""
    if value is None:
        return None
    s = value.replace('%', '').replace(' ', '').replace('.', '')  # Fjern punktum som tusindtalsseparator
    s = s.replace(',', '.')  # Erstat komma med punktum til float
    try:
        val = float(s) / 100
        return val
    except ValueError:
        return None

def simulate_paths(mu, sigma, weeks=520, simulations=100000):
    dt = 1 / 52
    returns = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt), size=(weeks, simulations))
    paths = np.exp(np.cumsum(returns, axis=0))
    paths = np.vstack([np.ones((1, simulations)), paths])
    return paths

def create_figures(paths, start_beløb=1_000_000):
    weeks = paths.shape[0] - 1
    mean = np.mean(paths, axis=1) * start_beløb
    p2_5 = np.percentile(paths, 2.5, axis=1) * start_beløb
    p97_5 = np.percentile(paths, 97.5, axis=1) * start_beløb

    årstal = np.arange(1, 11)
    u_index = np.array([a * 52 for a in årstal])
    vis_labels = np.array([1, 3, 5, 8, 10])
    vis_index = np.array([a * 52 for a in vis_labels])
    start_beløb_str = format_dk_tal(start_beløb) + " kr."

    # ---- GRAF 1 ----
    fig1, ax1 = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax1.fill_between(range(weeks + 1), p2_5, p97_5,
                     color="#d7c39d78", alpha=0.3,
                     label="95% Konfidensinterval")
    ax1.plot(mean, label="Forventet Afkast (Gennemsnit)",
             color="#dfcdabff", linestyle="--", linewidth=2)

    for år_lbl, idx in zip(vis_labels, vis_index):
        # Graf 1: 0 decimaler
        ax1.annotate(f"{mean[idx]:,.0f}".replace(',', '.') + " kr.", (idx, mean[idx]),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax1.annotate(f"{p97_5[idx]:,.0f}".replace(',', '.') + " kr.", (idx, p97_5[idx]),
                     textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
        ax1.annotate(f"{p2_5[idx]:,.0f}".replace(',', '.') + " kr.", (idx, p2_5[idx]),
                     textcoords="offset points", xytext=(0, -18), ha="center", fontsize=8)
        ax1.scatter(idx, mean[idx], marker="D", color="white", edgecolor="gray", zorder=5, s=40)
        ax1.scatter(idx, p97_5[idx], marker="D", color="white", edgecolor="gray", zorder=5, s=40)
        ax1.scatter(idx, p2_5[idx], marker="D", color="white", edgecolor="gray", zorder=5, s=40)

    # Aksemærker for alle år 1-10
    ax1.set_xticks(u_index)
    ax1.set_xticklabels([f"{a} år" for a in årstal])
    ax1.set_title("FORVENTET FORMUEUDVIKLING", pad=15, fontsize=20, fontweight='bold')
    ax1.set_xlabel("Tidshorisont", labelpad=15)
    ax1.set_ylabel(f"Porteføljeværdi (start = {start_beløb_str})", labelpad=15)
    ax1.yaxis.set_major_formatter(FuncFormatter(tusind_millioner_formatter))

    # Dynamisk y-akse med 10% luft (0 vises kun hvis relevant)
    y_min = min(p2_5.min(), mean.min())
    y_max = max(p97_5.max(), mean.max())
    y_range = y_max - y_min
    ax1.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # Ekstra luft til højre for år 10
    ax1.set_xlim(0, u_index[-1] + 0.5 * (u_index[1] - u_index[0]))

    ax1.legend(loc="upper left")
    ax1.grid(False)

    # ---- GRAF 2 ----
    år = vis_labels
    lower_afk = (p2_5[vis_index] / start_beløb) ** (1 / år) - 1
    upper_afk = (p97_5[vis_index] / start_beløb) ** (1 / år) - 1
    exp_afkast = (mean[vis_index] / start_beløb) ** (1 / år) - 1

    x = np.arange(len(år))
    fig2, ax2 = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax2.bar(x, (upper_afk - lower_afk) * 100, bottom=lower_afk * 100, width=0.4, color="#d7c39d78")

    for i in range(len(x)):
        # Graf 2: 2 decimaler
        exp_txt = f"{exp_afkast[i]*100:.2f}".replace('.', ',') + '%'
        ax2.scatter(x[i], exp_afkast[i]*100, color="white", edgecolors="gray",
                    marker="D", s=40, zorder=5)
        ax2.text(x[i], exp_afkast[i]*100 + 1.8, exp_txt,
                 ha="center", va="bottom", fontsize=9)
        upper_txt = f"{upper_afk[i]*100:.2f}".replace('.', ',') + '%'
        ax2.text(x[i], upper_afk[i]*100 + 1.5, upper_txt,
                 ha="center", va="bottom", fontsize=9)
        lower_txt = f"{lower_afk[i]*100:.2f}".replace('.', ',') + '%'
        ax2.text(x[i], lower_afk[i]*100 - 2, lower_txt,
                 ha="center", va="top", fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{a} år" for a in år])
    ax2.set_title("FORVENTET ÅRLIGT AFKAST", pad=15, fontsize=20, fontweight='bold')
    ax2.set_xlabel("Tidshorisont", labelpad=15)
    ax2.set_ylabel("Årligt afkast p.a. (%)", labelpad=15)

    # Dynamisk y-akse med 10% luft
    y2_min = min(lower_afk * 100)
    y2_max = max(upper_afk * 100)
    y2_range = y2_max - y2_min
    ax2.set_ylim(y2_min - 0.1 * y2_range, y2_max + 0.1 * y2_range)

    # Y-aksemærker uden decimaler
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(round(x)):,}".replace(",", ".") + '%'))

    ax2.grid(False)  # Ingen gridlines
    ax2.axhline(0, color="gray", linewidth=0.8)  # 0-akse

    return fig1, fig2

def fig_to_uri(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return "data:image/png;base64," + encoded

def generate_pdf(fig1, fig2):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=landscape(A4))
    pw, ph = landscape(A4)

    img_width = pw * 0.96
    img_height = ph * 0.67
    text_fontsize = 8
    text_gap = 15         # Afstand fra graf til EXPL
    bottom_margin = 120    # Afstand fra bund til graf (øges for mere luft)

    header_img_path = "assets/Header.png"
    footer_img_path = "assets/Footer.png"

    def draw_header_footer():
        # Header øverst til højre
        try:
            header_img = ImageReader(header_img_path)
            header_w, header_h = header_img.getSize()
            scale = 60 / header_h
            draw_w, draw_h = header_w * scale, header_h * scale
            c.drawImage(header_img, pw - draw_w - 30, ph - draw_h - 10, width=draw_w, height=draw_h)
        except Exception as e:
            print("Header fejl:", e)
        # Footer midtfor
        try:
            footer_img = ImageReader(footer_img_path)
            footer_w, footer_h = footer_img.getSize()
            scale = 35 / footer_h
            draw_w, draw_h = footer_w * scale, footer_h * scale
            c.drawImage(footer_img, (pw - draw_w) / 2, 10, width=draw_w, height=draw_h)
        except Exception as e:
            print("Footer fejl:", e)

    def draw_bottom_image(fig, bottom_margin=40):
        img_buf = BytesIO()
        fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
        img_buf.seek(0)
        img = ImageReader(img_buf)
        iw, ih = img.getSize()
        scale = min(img_width / iw, img_height / ih)
        draw_w, draw_h = iw * scale, ih * scale
        x = (pw - draw_w) / 2
        y = bottom_margin
        c.drawImage(img, x, y, width=draw_w, height=draw_h)
        return y  # Returnér bunden af billedet

    # --- Side 1: Graf 1 ---
    draw_header_footer()
    y_img_bottom = draw_bottom_image(fig1, bottom_margin=bottom_margin)
    c.setFont("Helvetica-Oblique", text_fontsize)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    expl1_lines = [
        "Forklaring: Grafen viser en simulering af porteføljeudvikling baseret på stokastiske ugentlige afkast over ti år.",
        "Afkastforventningerne er baseret på de nyeste vurderinger fra Rådet for Afkastforventninger."
    ]
    for i, line in enumerate(expl1_lines):
        c.drawCentredString(pw/2, y_img_bottom - text_gap - i*text_fontsize*1.3, line)

    c.showPage()

    # --- Side 2: Graf 2 ---
    draw_header_footer()
    y_img_bottom = draw_bottom_image(fig2, bottom_margin=bottom_margin)
    c.setFont("Helvetica-Oblique", text_fontsize)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    expl2_lines = [
        "Forklaring: Grafen viser det årlige forventede afkast og et konfidensinterval på 95 procent.",
        "Afkastforventningerne er baseret på estimater fra Rådet for Afkastforventninger."
    ]
    for i, line in enumerate(expl2_lines):
        c.drawCentredString(pw/2, y_img_bottom - text_gap - i*text_fontsize*1.3, line)

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Img(src="/assets/Header.png", style={"position": "absolute", "top": "20px", "right": "40px", "height": "120px"}),
    html.H1("Simulering Af Porteføljens Risiko", style={"textAlign": "center", "fontSize": "2.5rem"}),
    html.Br(),
    html.Label("Startbeløb i kr."),
    dcc.Input(id="start-beløb", type="text", value="1.000.000", style={"marginBottom": "20px", "marginLeft": "15px"}),
    html.Br(),
    html.Label("Forventet Afkast p.a."),
    dcc.Input(id="mu", type="text", value="6,00%", style={"marginBottom": "20px", "marginLeft": "15px"}),
    html.Br(),
    html.Label("Forventet Standardafvigelse p.a."),
    dcc.Input(id="sigma", type="text", value="9,00%", style={"marginBottom": "20px", "marginLeft": "15px"}),
    html.Br(),
    html.Button("Dan Risiko Rapport i PDF", id="pdf-btn"),
    html.A("Download PDF", id="download-link", href="", target="_blank", download="RisikoRapport.pdf", style={"display": "none", "marginLeft": "20px"}),
    html.Div(id="error-msg", style={"color": "red", "textAlign": "center"}),
    html.Div([
        html.Div([
            html.Img(id="fig1", style={"width": "1200px", "height": "700px", "display": "block", "margin": "0 auto"}),
            html.Div(id="expl1", style={"textAlign": "center", "color": "gray", "fontStyle": "italic", "whiteSpace": "pre-line", "marginTop": "10px"}),
        ]),
        html.Div(style={"height": "60px"}),  # Ekstra luft mellem graferne
        html.Div([
            html.Img(id="fig2", style={"width": "1200px", "height": "700px", "display": "block", "margin": "0 auto"}),
            html.Div(id="expl2", style={"textAlign": "center", "color": "gray", "fontStyle": "italic", "whiteSpace": "pre-line", "marginTop": "10px"}),
        ]),
    ]),
    html.Br(),
], style={"fontFamily": "Helvetica, Arial, sans-serif", "position": "relative"})

@app.callback(
    Output("fig1", "src"),
    Output("fig2", "src"),
    Output("expl1", "children"),
    Output("expl2", "children"),
    Output("error-msg", "children"),
    Input("start-beløb", "value"),
    Input("mu", "value"),
    Input("sigma", "value"),
)
def update_graph(start_str, mu_str, sigma_str):
    # Parse startbeløb - fjern evt. punktummer og parse til int
    try:
        start_renset = start_str.replace(".", "").replace(" ", "")
        start_beløb = int(start_renset)
    except Exception:
        return "", "", "", "", "Ugyldigt startbeløb. Brug format som fx 1.000.000"

    mu = parse_percent(mu_str)
    sigma = parse_percent(sigma_str)
    if mu is None or sigma is None:
        return "", "", "", "", "Ugyldigt format for afkast eller volatilitet. Brug fx 6,00% eller 18,00%"

    if mu < 0 or sigma <= 0:
        return "", "", "", "", "Afkast og volatilitet skal være positive tal."

    paths = simulate_paths(mu, sigma)
    fig1, fig2 = create_figures(paths, start_beløb=start_beløb)

    src1 = fig_to_uri(fig1)
    src2 = fig_to_uri(fig2)

    expl1 = "Forklaring: Grafen viser en simulering af porteføljeudvikling baseret på stokastiske ugentlige afkast over ti år.\nAfkastforventningerne er baseret på de nyeste vurderinger fra Rådet for Afkastforventninger."
    expl2 = "Forklaring: Grafen viser det årlige forventede afkast og et konfidensinterval på 95 procent.\nAfkastforventningerne er baseret på estimater fra Rådet for Afkastforventninger."

    return src1, src2, expl1, expl2, ""

@app.callback(
    Output("download-link", "href"),
    Output("download-link", "style"),
    Input("pdf-btn", "n_clicks"),
    State("start-beløb", "value"),
    State("mu", "value"),
    State("sigma", "value"),
    prevent_initial_call=True,
)
def generate_pdf_callback(n_clicks, start_str, mu_str, sigma_str):
    try:
        start_renset = start_str.replace(".", "").replace(" ", "")
        start_beløb = int(start_renset)
        mu = parse_percent(mu_str)
        sigma = parse_percent(sigma_str)
        if mu is None or sigma is None or mu < 0 or sigma <= 0:
            return "", {"display": "none"}
        paths = simulate_paths(mu, sigma)
        fig1, fig2 = create_figures(paths, start_beløb=start_beløb)
        pdf_data = generate_pdf(fig1, fig2)
        pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")
        href = f"data:application/pdf;base64,{pdf_base64}"
        return href, {"display": "inline", "marginLeft": "20px"}
    except Exception as e:
        print("PDF ERROR:", e)
        return "", {"display": "none"}

if __name__ == "__main__":
    app.run(debug=True)