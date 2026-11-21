"""Build the unified 7-slide MSDM 5054 Final Project presentation (.pptx).

Run via uv without modifying the project's core venv:
    uv run --with python-pptx --with pillow python scripts/build_presentation.py
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "outputs" / "figures"
OUT_PPTX = ROOT / "report" / "final_presentation.pptx"

# Clean academic palette (HKUST navy + slate + teal/amber accents)
NAVY = RGBColor(16, 42, 76)
DARK_SLATE = RGBColor(34, 48, 66)
MUTED = RGBColor(92, 107, 125)
TEAL = RGBColor(18, 116, 133)
AMBER = RGBColor(189, 119, 28)
LIGHT_BG = RGBColor(246, 248, 251)
CARD_BG = RGBColor(255, 255, 255)
BORDER = RGBColor(216, 224, 234)
ACCENT_BG = RGBColor(235, 243, 248)
WHITE = RGBColor(255, 255, 255)

FONT = "Calibri"


def set_bg(slide, color: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, tag: str, title: str, subtitle: str = "") -> None:
    # Top navy accent bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.10))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()

    # Tag pill
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(0.24), Inches(2.5), Inches(0.30))
    pill.fill.solid()
    pill.fill.fore_color.rgb = ACCENT_BG
    pill.line.color.rgb = BORDER
    pill.line.width = Pt(0.75)
    tf = pill.text_frame
    tf.word_wrap = False
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = tag
    r.font.name = FONT
    r.font.size = Pt(10.5)
    r.font.bold = True
    r.font.color.rgb = TEAL

    # Title
    tb = slide.shapes.add_textbox(Inches(0.55), Inches(0.58), Inches(12.2), Inches(0.52))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.name = FONT
    r.font.size = Pt(24)
    r.font.bold = True
    r.font.color.rgb = NAVY

    if subtitle:
        sb = slide.shapes.add_textbox(Inches(0.55), Inches(1.06), Inches(12.2), Inches(0.30))
        stf = sb.text_frame
        stf.word_wrap = True
        stf.margin_left = stf.margin_top = stf.margin_right = stf.margin_bottom = 0
        sp = stf.paragraphs[0]
        sr = sp.add_run()
        sr.text = subtitle
        sr.font.name = FONT
        sr.font.size = Pt(12.5)
        sr.font.color.rgb = MUTED


def add_footer(slide, page_num: int, total: int = 7) -> None:
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(7.10), Inches(12.23), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER
    line.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(0.55), Inches(7.14), Inches(9.5), Inches(0.28))
    tf = tb.text_frame
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "MSDM 5054 Statistical Machine Learning  |  Deconstructing Visual Alpha (HKUST, Fall 2026)"
    r.font.name = FONT
    r.font.size = Pt(9.5)
    r.font.color.rgb = MUTED

    pb = slide.shapes.add_textbox(Inches(11.5), Inches(7.14), Inches(1.28), Inches(0.28))
    ptf = pb.text_frame
    ptf.margin_left = ptf.margin_top = ptf.margin_right = ptf.margin_bottom = 0
    pp = ptf.paragraphs[0]
    pp.alignment = PP_ALIGN.RIGHT
    pr = pp.add_run()
    pr.text = f"{page_num} / {total}"
    pr.font.name = FONT
    pr.font.size = Pt(9.5)
    pr.font.bold = True
    pr.font.color.rgb = NAVY


def add_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str = "",
    bullets: list[str] | None = None,
    bg: RGBColor = CARD_BG,
    border: RGBColor = BORDER,
    title_color: RGBColor = NAVY,
    font_size: float = 12.0,
) -> None:
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.adjustments[0] = 0.04
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg
    shape.line.color.rgb = border
    shape.line.width = Pt(1.0)

    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.16)
    tf.margin_right = Inches(0.16)
    tf.margin_top = Inches(0.11)
    tf.margin_bottom = Inches(0.10)

    first = True
    if title:
        p = tf.paragraphs[0]
        first = False
        p.space_after = Pt(4)
        r = p.add_run()
        r.text = title
        r.font.name = FONT
        r.font.size = Pt(font_size + 1.5)
        r.font.bold = True
        r.font.color.rgb = title_color

    for b in bullets or []:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(3.5)
        r = p.add_run()
        r.text = f"•  {b}"
        r.font.name = FONT
        r.font.size = Pt(font_size)
        r.font.color.rgb = DARK_SLATE


def add_picture_fit(slide, img_path: Path, left: float, top: float, max_w: float, max_h: float) -> None:
    with Image.open(img_path) as im:
        iw, ih = im.size
    ratio = iw / ih
    if max_w / max_h > ratio:
        h = max_h
        w = h * ratio
    else:
        w = max_w
        h = w / ratio
    x = left + (max_w - w) / 2.0
    y = top + (max_h - h) / 2.0
    slide.shapes.add_picture(str(img_path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def set_notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text.strip()


def build() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank)
    set_bg(s1, NAVY)

    # Decorative accent strip
    strip = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.28), Inches(7.5))
    strip.fill.solid()
    strip.fill.fore_color.rgb = TEAL
    strip.line.fill.background()

    # Course tag
    tb = s1.shapes.add_textbox(Inches(0.85), Inches(0.85), Inches(11.6), Inches(0.4))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = "MSDM 5054 STATISTICAL MACHINE LEARNING  ·  FINAL PROJECT PRESENTATION (FALL 2026)"
    r.font.name = FONT
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = RGBColor(148, 210, 224)

    # Main Title
    tb = s1.shapes.add_textbox(Inches(0.85), Inches(1.45), Inches(11.6), Inches(1.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "Deconstructing Visual Alpha"
    r.font.name = FONT
    r.font.size = Pt(44)
    r.font.bold = True
    r.font.color.rgb = WHITE

    p2 = tf.add_paragraph()
    p2.space_before = Pt(8)
    r2 = p2.add_run()
    r2.text = "An Empirical Audit of Inductive Bias, Label Noise, Calibration, and Concept Drift in Financial Chart CNNs"
    r2.font.name = FONT
    r2.font.size = Pt(20)
    r2.font.color.rgb = RGBColor(214, 228, 242)

    # Four pillars bar
    pillars = [
        ("RQ1 · Inductive Bias", "Minyi Xie\n2D-CNN vs. ML & Masking"),
        ("RQ2 · Label Noise", "Dongqing Zhao\nMargin Filter & Horizons"),
        ("RQ3 · Calibration", "Hanchong Qiu\nEnsembling & Tail Accuracy"),
        ("RQ4 · Drift & CAM", "Ziyuan Yan\n2006–2012 & Grad-CAM"),
    ]
    for i, (head, body) in enumerate(pillars):
        x = 0.85 + i * 2.95
        box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(3.85), Inches(2.75), Inches(1.30))
        box.adjustments[0] = 0.08
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(24, 58, 99)
        box.line.color.rgb = RGBColor(58, 108, 152)
        box.line.width = Pt(1.0)
        btf = box.text_frame
        btf.word_wrap = True
        btf.margin_left = Inches(0.14)
        btf.margin_top = Inches(0.14)
        bp1 = btf.paragraphs[0]
        br1 = bp1.add_run()
        br1.text = head
        br1.font.name = FONT
        br1.font.size = Pt(13)
        br1.font.bold = True
        br1.font.color.rgb = RGBColor(148, 210, 224)
        bp2 = btf.add_paragraph()
        bp2.space_before = Pt(5)
        br2 = bp2.add_run()
        br2.text = body
        br2.font.name = FONT
        br2.font.size = Pt(11.5)
        br2.font.color.rgb = WHITE

    # Authors & Affiliation
    ab = s1.shapes.add_textbox(Inches(0.85), Inches(5.65), Inches(11.8), Inches(1.2))
    atf = ab.text_frame
    atf.word_wrap = True
    ap1 = atf.paragraphs[0]
    ar1 = ap1.add_run()
    ar1.text = "Hanchong Qiu (21356531)   ·   Minyi Xie (21319480)   ·   Dongqing Zhao (21420966)   ·   Ziyuan Yan (21219496)"
    ar1.font.name = FONT
    ar1.font.size = Pt(14)
    ar1.font.bold = True
    ar1.font.color.rgb = WHITE

    ap2 = atf.add_paragraph()
    ap2.space_before = Pt(6)
    ar2 = ap2.add_run()
    ar2.text = "Department of Mathematics / Data-Driven Modeling, HKUST   |   Replication & Audit of Jiang, Kelly, & Xiu (JF, 2023)"
    ar2.font.name = FONT
    ar2.font.size = Pt(12)
    ar2.font.color.rgb = RGBColor(186, 206, 226)

    set_notes(
        s1,
        "[0:00-0:20] Hello everyone. Today we present 'Deconstructing Visual Alpha', a four-stage statistical machine learning audit "
        "of Jiang, Kelly, and Xiu's 2023 Journal of Finance study on predicting stock returns from chart images. "
        "Instead of treating the CNN as a black box, our team—Hanchong Qiu, Minyi Xie, Dongqing Zhao, and Ziyuan Yan—audits "
        "inductive bias, label noise, probability calibration, and temporal drift on a single unified benchmark.",
    )

    # =========================================================================
    # SLIDE 2: MOTIVATION, DATASET & BASELINE REPLICATION
    # =========================================================================
    s2 = prs.slides.add_slide(blank)
    set_bg(s2, LIGHT_BG)
    add_header(
        s2,
        "SETUP & BASELINE",
        "Unified Benchmark & Baseline CNN Replication (1993–2019)",
        "Strict temporal split, memory-mapped 64×60 binary OHLC+MA+Volume charts, and 4-metric evaluation contract",
    )

    add_card(
        s2,
        0.55,
        1.52,
        5.95,
        5.30,
        "Data Protocol & Architectural Setup",
        [
            "Dataset: 2,196,994 U.S. equity monthly 20-day chart images (64×60 uint8 binary: rows 0–50 OHLC+MA, row 51 black separator, rows 52–63 volume bars).",
            "Label vs. Return Alignment: Binary target Retx_20d_label uses ex-dividend return (drop missing code 2 → 2,181,781 valid); portfolio returns use total return Ret_20d.",
            "Strict Temporal Split: Train/Val on 1993–1999 (482,926 / 206,968, 70/30 seeded); 20-year untouched OOS test on 2000–2019 (1,491,887 stock-months).",
            "Baseline CNN20D (708,866 params): 3 asymmetric (5×3) Conv blocks (1→64→128→256) + BN + LeakyReLU + vertical (2×1) MaxPool + 50% Dropout; Adam lr=1e-5, early stopping.",
            "Memory Safety: Read-only np.memmap with on-device uint8→float32/255 scaling.",
        ],
        font_size=12.0,
    )

    # Right top KPI strip
    kpis = [
        ("52.50%", "Ensemble OOS Acc"),
        ("0.5355", "Ensemble ROC-AUC"),
        ("1.68", "L–S Sharpe (t=7.51)"),
        ("+22.33%", "Annual L–S Spread"),
    ]
    for i, (val, lbl) in enumerate(kpis):
        x = 6.75 + i * 1.54
        c = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.52), Inches(1.44), Inches(1.05))
        c.adjustments[0] = 0.08
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY if i == 2 else CARD_BG
        c.line.color.rgb = NAVY if i == 2 else BORDER
        tf = c.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p1 = tf.paragraphs[0]
        p1.alignment = PP_ALIGN.CENTER
        r1 = p1.add_run()
        r1.text = val
        r1.font.name = FONT
        r1.font.size = Pt(18)
        r1.font.bold = True
        r1.font.color.rgb = WHITE if i == 2 else TEAL
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = lbl
        r2.font.name = FONT
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = RGBColor(210, 225, 240) if i == 2 else MUTED

    # Right middle table
    rows, cols = 4, 6
    tbl_shape = s2.shapes.add_table(rows, cols, Inches(6.75), Inches(2.75), Inches(6.03), Inches(2.05))
    tbl = tbl_shape.table
    headers = ["Model (2000–2019 OOS)", "Acc (%)", "CE Loss", "AUC", "ECE ↓", "L–S Sharpe"]
    data = [
        ["CNN20D (seed=42)", "52.33", "0.6924", "0.5322", "0.0166", "1.70 (t=7.59)"],
        ["Single CNN (5-seed μ±σ)", "52.09±0.30", "0.6933", "0.5301", "0.0215", "1.74±0.19"],
        ["5-Seed Deep Ensemble", "52.50", "0.6915", "0.5355", "0.0105", "1.68 (t=7.51)"],
    ]
    col_widths = [2.25, 0.82, 0.72, 0.72, 0.68, 0.84]
    for ci, w in enumerate(col_widths):
        tbl.columns[ci].width = Inches(w)
    for ci, h in enumerate(headers):
        cell = tbl.cell(0, ci)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
        r = p.add_run()
        r.text = h
        r.font.name = FONT
        r.font.size = Pt(10)
        r.font.bold = True
        r.font.color.rgb = WHITE
    for ri, row in enumerate(data, start=1):
        is_ens = ri == 3
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT_BG if is_ens else WHITE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = val
            r.font.name = FONT
            r.font.size = Pt(10)
            r.font.bold = is_ens
            r.font.color.rgb = NAVY if is_ens else DARK_SLATE

    add_card(
        s2,
        6.75,
        4.98,
        6.03,
        1.84,
        "Why Audit Instead of Chasing Accuracy?",
        [
            "Financial return classification operates near the Bayes bound (Acc ≈ 52%), yet yields Sharpe ~1.7.",
            "Our 4 RQs dissect: (1) where the spatial signal lives, (2) whether near-zero labels are noise, (3) how calibration & tail selection create Sharpe, and (4) whether signals survive into 2013–2019.",
        ],
        bg=ACCENT_BG,
        font_size=11.5,
    )
    add_footer(s2, 2)
    set_notes(
        s2,
        "[0:20-1:00] Our benchmark spans 27 years of U.S. equities—nearly 2.2 million 64-by-60 binary charts containing 20-day OHLC bars, "
        "a moving average, and bottom volume bars. Following Jiang et al., we train strictly on 1993 to 1999 and hold out 1.49 million "
        "stock-months from 2000 to 2019 untouched. We also enforced a strict schema audit separating ex-dividend classification labels from "
        "dividend-inclusive portfolio returns. Our 5-seed replication confirms that Visual Alpha is real: individual seeds average 1.74 Sharpe, "
        "and the 5-seed ensemble reaches 52.50% accuracy, 0.5355 AUC, and a 22.33% annualized long-short spread with t-stat 7.51. "
        "Next, we audit four statistical ML questions along the modeling pipeline.",
    )

    # =========================================================================
    # SLIDE 3: RQ1 — INDUCTIVE BIAS & SPATIAL MASKING (MINYI XIE)
    # =========================================================================
    s3 = prs.slides.add_slide(blank)
    set_bg(s3, LIGHT_BG)
    add_header(
        s3,
        "RQ1 · MINYI XIE",
        "Inductive Bias & Visual Component Masking",
        "Does predictive power require 2D local convolutions, and which chart regions carry the signal?",
    )

    add_card(
        s3,
        0.55,
        1.48,
        12.23,
        1.05,
        "Key Finding: Compact 2D Spatial Inductive Bias Wins; Recent 10 Days Dominate Signal",
        [
            "2D-CNN (seed 42: AUC 0.5322, Sharpe 1.70) beats Temporal 1D-CNN (0.5254 / 1.49), PCA+LR (0.5237 / 1.52), and deeper ResNet (0.5252 / 0.98). Pixel-summary LightGBM matches AUC (0.5317) but lags sharply in tail Sharpe (1.18).",
        ],
        bg=ACCENT_BG,
        font_size=11.5,
    )

    # Figure 1 frame
    frame1 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(2.64), Inches(12.23), Inches(3.22))
    frame1.adjustments[0] = 0.02
    frame1.fill.solid()
    frame1.fill.fore_color.rgb = WHITE
    frame1.line.color.rgb = BORDER
    add_picture_fit(s3, FIGS / "fig1_sq1_ablation.png", 0.65, 2.70, 12.03, 3.10)

    add_card(
        s3,
        0.55,
        5.96,
        12.23,
        1.02,
        "",
        [
            "Temporal Asymmetry: Masking the first 10 days leaves AUC intact (0.5332), whereas masking the most recent 10 days collapses AUC to 0.5031 and Sharpe to 0.34.",
            "Volume Contribution: Masking volume at inference cuts accuracy (-0.79 pp) and Sharpe (-0.21); retraining without volume recovers AUC to 0.5270 with a smaller spread (15.61% vs. 21.32%) and lower volatility (8.52% vs. 12.57%, Sharpe 1.83).",
        ],
        font_size=11.0,
    )
    add_footer(s3, 3)
    set_notes(
        s3,
        "[1:00-1:45] In RQ1, led by Minyi Xie, we hold the pixels fixed and vary the model family. As shown on the left of Figure 1, "
        "the compact 2D-CNN leads the temporal 1D-CNN by 0.0068 AUC and 0.21 Sharpe, and also outperforms PCA+Logistic Regression and a larger ResNet—"
        "confirming that higher capacity hurts under low SNR. Interestingly, LightGBM on hand-crafted pixel summaries nearly matches overall AUC at 0.5317, "
        "yet its long-short Sharpe drops to 1.18 versus 1.70, showing that global ranking and tail ranking differ. "
        "On the right, inference-time occlusion reveals sharp temporal asymmetry: hiding the first 10 days preserves AUC at 0.5332, "
        "but hiding the recent 10 days collapses AUC to 0.5031 and Sharpe to 0.34. Hiding volume also degrades performance, and retraining without volume "
        "never fully recovers baseline AUC.",
    )

    # =========================================================================
    # SLIDE 4: RQ2 — LABEL NOISE, MARGIN FILTERING & HORIZONS (DONGQING ZHAO)
    # =========================================================================
    s4 = prs.slides.add_slide(blank)
    set_bg(s4, LIGHT_BG)
    add_header(
        s4,
        "RQ2 · DONGQING ZHAO",
        "Supervision Noise, Margin Filtering & Forecast Horizons",
        "Are near-zero return labels disposable noise, or load-bearing for low-SNR probability estimation?",
    )

    add_card(
        s4,
        0.55,
        1.48,
        12.23,
        1.05,
        "Key Finding: Margin Filtering Hurts Monotonically; Label Smoothing Calibrates for Free",
        [
            "Dropping training charts with |Retx_20d| < 5% cuts test AUC 0.5322→0.5170 (>4× seed noise floor) and inflates ECE 4.7× (0.0166→0.0786). Keeping all charts with label smoothing (ε=0.1) cuts ECE 17% (0.0166→0.0137) at zero ranking cost.",
        ],
        bg=ACCENT_BG,
        font_size=11.5,
    )

    frame2 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(2.64), Inches(12.23), Inches(3.22))
    frame2.adjustments[0] = 0.02
    frame2.fill.solid()
    frame2.fill.fore_color.rgb = WHITE
    frame2.line.color.rgb = BORDER
    add_picture_fit(s4, FIGS / "fig2_sq2_noise_horizon.png", 0.65, 2.70, 12.03, 3.10)

    add_card(
        s4,
        0.55,
        5.96,
        12.23,
        1.02,
        "",
        [
            "Not Mere Sample-Size Loss: Standardized |t|≥0.5 keeps 94.4% of data yet loses more AUC (0.5283) than τ=1% (89.0% kept, 0.5303). Paired monthly spread C2−C0 = −0.37 pp/mo (95% CI [−0.60, −0.18]), exposing what level Sharpe hides.",
            "Cross-Horizon Transfer: On 1,471,746 matched charts, AUC is flat across 5/20/60d (0.5298 / 0.5315 / 0.5296); raw accuracy rises purely from base-rate drift (up-rate 50.46%→51.51%), while economics peak at the native 20-day horizon.",
        ],
        font_size=11.0,
    )
    add_footer(s4, 4)
    set_notes(
        s4,
        "[1:45-2:30] In RQ2, led by Dongqing Zhao, we test a natural denoising hypothesis: since +0.01% and -0.01% returns get opposite binary labels, "
        "does filtering out near-zero training returns help? The answer is a clear no. As threshold tau increases from 0% to 5%, test AUC falls monotonically "
        "from 0.5322 to 0.5170 and ECE explodes nearly fivefold. Even at tau=3%, where level Sharpe looks flat, paired monthly spread differences are "
        "significantly negative at -0.37 percentage points per month. Near-zero observations are load-bearing: they anchor the conditional probability structure "
        "around 0.5. By contrast, keeping all charts and applying label smoothing (epsilon=0.1) reduces ECE by 17% for free with no AUC loss. "
        "Finally, relabeling the same 20-day charts for 5- and 60-day horizons yields virtually identical AUC, showing predictability belongs to the chart itself.",
    )

    # =========================================================================
    # SLIDE 5: RQ3 — CALIBRATION & SELECTIVE PREDICTION (HANCHONG QIU)
    # =========================================================================
    s5 = prs.slides.add_slide(blank)
    set_bg(s5, LIGHT_BG)
    add_header(
        s5,
        "RQ3 · HANCHONG QIU",
        "Probability Calibration, Deep Ensembling & Selective Prediction",
        "How does a 52.5% accuracy classifier produce a 1.7 Sharpe ratio, and how should we calibrate it?",
    )

    add_card(
        s5,
        0.55,
        1.48,
        12.23,
        1.05,
        "Key Finding: Ensembling Cuts ECE by 51% Without Shift Fragility; Tail Accuracy Reaches 57.4%",
        [
            ">99% of OOS probabilities lie in [0.36, 0.64]. 5-seed ensembling cuts ECE 0.0215→0.0105 (−51.2%) and raises AUC 0.5322→0.5355; post-hoc temperature scaling on the ensemble (T=0.878) over-sharpens on OOS (ECE 0.0105→0.0155).",
        ],
        bg=ACCENT_BG,
        font_size=11.5,
    )

    frame3 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(2.64), Inches(12.23), Inches(3.22))
    frame3.adjustments[0] = 0.02
    frame3.fill.solid()
    frame3.fill.fore_color.rgb = WHITE
    frame3.line.color.rgb = BORDER
    add_picture_fit(s5, FIGS / "fig3_sq3_calibration.png", 0.65, 2.70, 12.03, 3.10)

    add_card(
        s5,
        0.55,
        5.96,
        12.23,
        1.02,
        "",
        [
            "Non-Stationary Calibration Pitfall: Parametric temperature scaling fitted on higher-SNR 1993–1999 validation data (T<1) hurts calibration in lower-SNR 2000–2019 markets, whereas non-parametric ensembling stays well-calibrated.",
            "Selective Prediction Solves the Sharpe Puzzle: Ranking by confidence margin |p̂−0.5|, ensemble accuracy climbs monotonically from 52.50% (100% coverage) → 55.48% (top 20%, i.e., D10+D1 deciles) → 57.40% (top 5%, AUC 0.5770).",
        ],
        font_size=11.0,
    )
    add_footer(s5, 5)
    set_notes(
        s5,
        "[2:30-3:15] In RQ3, I examine probability calibration and selective prediction. On the left of Figure 3, notice that over 99% of out-of-sample "
        "predictions sit inside the narrow band from 0.36 to 0.64. A single CNN is mildly overconfident, so validation temperature scaling (T=1.064) "
        "reduces single-model ECE by 14.5%. However, 5-seed deep ensembling is far more effective: it cuts ECE by 51.2% down to 0.0105 while also "
        "boosting AUC to 0.5355. Strikingly, stacking temperature scaling on top of the ensemble actually worsens OOS calibration because 1990s validation "
        "had higher SNR and fitted T=0.878, over-sharpening 2000–2019 forecasts. On the right, selective prediction resolves the central puzzle of Visual Alpha: "
        "abstaining on uncertain stocks raises ensemble accuracy monotonically from 52.50% overall to 55.48% in the top-20% decile tails and 57.40% in the top 5%.",
    )

    # =========================================================================
    # SLIDE 6: RQ4 — TEMPORAL TRANSFER & GRAD-CAM (ZIYUAN YAN)
    # =========================================================================
    s6 = prs.slides.add_slide(blank)
    set_bg(s6, LIGHT_BG)
    add_header(
        s6,
        "RQ4 · ZIYUAN YAN",
        "Temporal Transfer (1990s vs. 2006–2012) & Grad-CAM Attribution",
        "Does retraining on recent market eras fix decade-level decay, and where does CNN20D focus?",
    )

    # Left: Figure 4 (large and crisp)
    frame4 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(1.48), Inches(7.70), Inches(5.45))
    frame4.adjustments[0] = 0.02
    frame4.fill.solid()
    frame4.fill.fore_color.rgb = WHITE
    frame4.line.color.rgb = BORDER
    add_picture_fit(s6, FIGS / "fig4_sq4_drift_gradcam.png", 0.62, 1.54, 7.56, 5.33)

    # Right: 3 stacked analytical cards
    add_card(
        s6,
        8.42,
        1.48,
        4.36,
        1.55,
        "1. Decade-Level Decay (2000–2019)",
        [
            "1990s single CNN pooled AUC falls 0.5394 (2000s) → 0.5218 (2010s); 5-seed ensemble falls 0.5447 → 0.5223.",
            "Monthly equal-weighted AUC diff: −0.00899 (95% 3-mo block-bootstrap CI [−0.01613, −0.00155]).",
        ],
        font_size=10.8,
    )

    add_card(
        s6,
        8.42,
        3.15,
        4.36,
        1.95,
        "2. 2006–2012 Retraining Fails to Help",
        [
            "Matched 2013–2019 holdout (N = 472,357): Modern AUC 0.5222 vs. 1990s 0.5217 (+0.00050 pooled).",
            "Paired 84-month AUC diff: −0.00215 (95% CI [−0.00850, +0.00447], includes zero).",
            "Probability quality worsens: ECE 0.0227 → 0.0606, CE 0.6939 → 0.7048, gross Sharpe 1.50 → 1.17.",
        ],
        bg=ACCENT_BG,
        font_size=10.8,
    )

    add_card(
        s6,
        8.42,
        5.22,
        4.36,
        1.71,
        "3. Global Top-1000 Grad-CAM",
        [
            "Block-3 Grad-CAM on 1,000 extreme UP vs. 1,000 extreme DOWN charts (parity MAE 9.9×10⁻⁸).",
            "Volume-band mass (rows 52–63): 20.27% for DOWN vs. 6.86% for UP (−13.41 pp, CI [−14.20, −12.68]).",
            "Post-hoc spatial attribution, not causal proof.",
        ],
        font_size=10.8,
    )
    add_footer(s6, 6)
    set_notes(
        s6,
        "[3:15-4:05] Finally, in RQ4, Ziyuan Yan investigates temporal transfer and visual attribution. For the frozen 1993–1999 CNN, "
        "pooled AUC declines from 0.5394 in the 2000s to 0.5218 in the 2010s, with a statistically negative monthly bootstrap difference. "
        "Surprisingly, retraining the exact same CNN on 2006–2012 does not fix this on the matched 2013–2019 holdout of 472,357 stock-months: "
        "pooled AUC edges up by only 0.0005, the 84-month paired AUC difference includes zero, and calibration worsens sharply—ECE nearly triples "
        "from 0.0227 to 0.0606 while gross Sharpe drops from 1.50 to 1.17. Meanwhile, running Grad-CAM on the 1,000 most confident UP and 1,000 "
        "most confident DOWN forecasts reveals a striking class-conditional asymmetry: predicted-DOWN charts place 20.27% of attribution mass "
        "in the volume band versus only 6.86% for predicted-UP charts. We carefully interpret this as post-hoc model attribution rather than causal market mechanics.",
    )

    # =========================================================================
    # SLIDE 7: SYNTHESIS, LIMITATIONS & TEAM CONTRIBUTIONS
    # =========================================================================
    s7 = prs.slides.add_slide(blank)
    set_bg(s7, LIGHT_BG)
    add_header(
        s7,
        "SYNTHESIS & CONCLUSION",
        "What Truly Drives Visual Alpha — and Where Are Its Limits?",
        "Takeaways across the four-stage statistical ML audit, key limitations, and full disclosures",
    )

    grid = [
        ("RQ1 · Inductive Bias (Minyi Xie)", "Compact asymmetric 2D-CNN beats 1D-CNN, linear, tree, and deeper ResNet in tail Sharpe; recent 10 days & volume carry the core signal.", TEAL),
        ("RQ2 · Supervision (Dongqing Zhao)", "Near-zero returns are load-bearing—margin filtering degrades AUC & ECE, while label smoothing improves calibration at zero ranking cost.", NAVY),
        ("RQ3 · Calibration (Hanchong Qiu)", "5-seed ensembling halves ECE without regime-shift fragility; selective tail accuracy (57.4% at top 5%) explains high decile Sharpe.", TEAL),
        ("RQ4 · Drift & CAM (Ziyuan Yan)", "Historical discrimination decays in the 2010s, yet 2006–2012 retraining fails to improve 2013–2019; Grad-CAM shows asymmetric volume focus.", AMBER),
    ]
    for i, (t, body, col) in enumerate(grid):
        r_idx, c_idx = divmod(i, 2)
        x = 0.55 + c_idx * 6.22
        y = 1.50 + r_idx * 1.48
        add_card(s7, x, y, 6.01, 1.34, t, [body], title_color=col, font_size=11.5)

    add_card(
        s7,
        0.55,
        4.55,
        5.90,
        2.38,
        "Limitations & Future Directions",
        [
            "Gross equal-weighted decile spreads omit bid–ask spreads, market impact, and turnover costs.",
            "Monthly snapshot rebalancing; single-seed runs for SQ1/SQ2/SQ4 modern retraining (no combined-era model).",
            "Future work: turnover-penalized portfolio optimization and multi-seed rolling retraining.",
        ],
        font_size=11.0,
    )

    add_card(
        s7,
        6.65,
        4.55,
        6.13,
        2.38,
        "Reproducibility & AI/LLM Usage Disclosure",
        [
            "Full code & 8-page report: github.com/hanchongqiu/Final_Project",
            "AI Disclosure: GitHub Copilot (Hanchong Qiu), OpenAI Codex (Minyi Xie & Dongqing Zhao), OpenCode GLM-5.3 (Dongqing Zhao), and ChatGPT (Ziyuan Yan) assisted with coding, debugging, plotting, and text/slide polishing.",
            "All training runs, schema checks, metrics, and interpretations were independently verified by the authors.",
        ],
        bg=ACCENT_BG,
        font_size=10.8,
    )
    add_footer(s7, 7)
    set_notes(
        s7,
        "[4:05-4:40] To summarize our four-stage audit: Visual Alpha is statistically real, but governed by tight low-SNR constraints. "
        "First, compact 2D convolutions over recent prices and volume beat both simpler and deeper alternatives in portfolio Sharpe. "
        "Second, near-zero returns should be kept and smoothed rather than discarded. Third, deep ensembling and selective tail prediction "
        "explain how a 52.5% classifier achieves a 1.7 Sharpe without post-hoc calibration fragility. Fourth, temporal generalization remains hard: "
        "newer training data alone did not beat the 1990s model, and Grad-CAM reveals asymmetric volume focus for bearish predictions. "
        "Key limitations include gross equal-weighted returns without transaction costs. All code, tests, and full AI usage disclosures are documented "
        "in our GitHub repository and 8-page report. Thank you for listening!",
    )

    OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PPTX))
    print(f"Saved presentation to {OUT_PPTX} ({OUT_PPTX.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
