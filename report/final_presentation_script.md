# MSDM 5054 Final Project — 5-Minute Presentation Script

- **Deck**: `report/final_presentation.pptx` (7 slides, 16:9 widescreen)
- **Target Duration**: **4 minutes 35 seconds** (~650 words at 140–145 wpm — safely under the strict **5:00** limit)
- **Team**: Hanchong Qiu (21356531), Minyi Xie (21319480), Dongqing Zhao (21420966), Ziyuan Yan (21219496)

---

## Slide 1 — Title & Overview (`0:00 – 0:20`, ~20s)
> Hello everyone. Today we present **"Deconstructing Visual Alpha"**, a four-stage statistical machine learning audit of Jiang, Kelly, and Xiu's 2023 *Journal of Finance* paper on predicting stock returns from chart images. Instead of treating the CNN as a black box, our team—Hanchong Qiu, Minyi Xie, Dongqing Zhao, and Ziyuan Yan—audits inductive bias, label noise, probability calibration, and temporal drift on one unified benchmark.

---

## Slide 2 — Unified Benchmark & Baseline Replication (`0:20 – 1:00`, ~40s)
> Our benchmark spans 27 years of U.S. equities—nearly 2.2 million 64-by-60 binary charts containing 20-day OHLC price bars, a moving average, and bottom volume bars. Following Jiang et al., we train strictly on 1993 to 1999 and hold out 1.49 million stock-months from 2000 to 2019 untouched. We also enforced a strict schema audit separating ex-dividend classification labels from dividend-inclusive portfolio returns.
>
> Our 5-seed replication confirms that Visual Alpha is real: individual seeds average a 1.74 Sharpe ratio, and the 5-seed ensemble achieves 52.50% accuracy, 0.5355 AUC, and a 22.33% annualized long–short spread with a *t*-statistic of 7.51. Next, we dissect why this works across four research questions.

---

## Slide 3 — RQ1: Inductive Bias & Visual Component Masking (`1:00 – 1:45`, ~45s)
> In RQ1, conducted by Minyi Xie, we hold the chart pixels fixed and vary the model family. On the left of Figure 1, the compact 2D-CNN outperforms the temporal 1D-CNN by 0.0068 AUC and 0.21 Sharpe, and also beats PCA plus logistic regression and a deeper ResNet—showing that excess capacity hurts under low signal-to-noise ratios. Notably, LightGBM on hand-crafted pixel summaries nearly matches overall AUC at 0.5317, yet its long–short Sharpe drops to 1.18 versus 1.70, highlighting the gap between global ranking and portfolio-tail ranking.
>
> On the right, inference-time occlusion reveals sharp temporal asymmetry: hiding the first 10 days preserves AUC at 0.5332, whereas hiding the most recent 10 days collapses AUC to 0.5031 and Sharpe to 0.34. Masking volume also degrades performance, and retraining without volume never recovers baseline AUC.

---

## Slide 4 — RQ2: Supervision Noise, Margin Filtering & Horizons (`1:45 – 2:30`, ~45s)
> In RQ2, conducted by Dongqing Zhao, we test a natural denoising hypothesis: since plus 0.01% and minus 0.01% returns receive opposite binary labels, does removing near-zero training returns help? The answer is a clear no. As the margin threshold rises from 0% to 5%, test AUC falls monotonically from 0.5322 to 0.5170 and calibration error inflates nearly fivefold. Even at 3%, where level Sharpe appears flat, paired monthly spread differences are significantly negative at minus 0.37 percentage points per month. Near-zero observations are load-bearing—they anchor conditional probabilities near 0.5.
>
> By contrast, keeping all charts and applying label smoothing ($\varepsilon = 0.1$) cuts ECE by 17% for free with zero AUC loss. Moreover, relabeling the same 20-day charts for 5- and 60-day horizons gives nearly identical AUC, showing predictability belongs to the chart itself.

---

## Slide 5 — RQ3: Probability Calibration & Selective Prediction (`2:30 – 3:15`, ~45s)
> In RQ3, I examine probability calibration and selective prediction. On the left of Figure 3, over 99% of out-of-sample predictions lie inside the narrow band from 0.36 to 0.64. A single CNN is mildly overconfident, so validation temperature scaling ($T = 1.064$) trims single-model ECE by 14.5%. However, 5-seed deep ensembling is far stronger: it cuts ECE by 51.2% down to 0.0105 while lifting AUC to 0.5355. Strikingly, stacking temperature scaling on top of the ensemble worsens OOS calibration, because higher-SNR 1990s validation data fits $T = 0.878$ and over-sharpens 2000–2019 forecasts.
>
> On the right, selective prediction resolves the central puzzle of Visual Alpha: abstaining on uncertain stocks raises ensemble accuracy monotonically from 52.50% overall to 55.48% in the top-20% decile tails and 57.40% in the top 5%.

---

## Slide 6 — RQ4: Temporal Transfer & Grad-CAM Attribution (`3:15 – 4:05`, ~50s)
> Finally, in RQ4, Ziyuan Yan tests temporal transfer and visual attribution. For the frozen 1993–1999 CNN, pooled AUC drops from 0.5394 in the 2000s to 0.5218 in the 2010s, with a statistically negative monthly bootstrap difference. Surprisingly, retraining the exact same CNN on 2006–2012 does not fix this on the matched 2013–2019 holdout of 472,357 stock-months: pooled AUC rises by only 0.0005, the 84-month paired AUC difference includes zero, and probability quality degrades sharply—ECE nearly triples from 0.0227 to 0.0606 while gross Sharpe falls from 1.50 to 1.17.
>
> Meanwhile, Block-3 Grad-CAM across the 1,000 strongest predicted-UP and 1,000 strongest predicted-DOWN charts uncovers a striking asymmetry: predicted-DOWN charts place 20.27% of attribution mass in the volume band versus only 6.86% for predicted-UP charts. We interpret this cautiously as post-hoc spatial attribution, not causal market evidence.

---

## Slide 7 — Synthesis, Limitations & Disclosure (`4:05 – 4:40`, ~35s)
> To wrap up: Visual Alpha is statistically real, but tightly governed by low-SNR constraints. Compact 2D convolutions over recent prices and volume outperform alternative families in tail Sharpe; near-zero labels should be smoothed rather than discarded; deep ensembling and selective tail prediction bridge 52.5% accuracy to a 1.7 Sharpe; and recent-era retraining alone does not reliably overcome temporal decay.
>
> Key limitations include gross equal-weighted returns without transaction costs and monthly snapshot rebalancing. All reproducible code, our 8-page report, and full AI usage disclosures are available on GitHub. Thank you for watching!

---

## Recording Tips (Keeping Strictly Under 5:00)
- **Word count**: ~665 words → **4:25–4:45** at natural speaking pace (~145 wpm).
- **Pacing cue**: Keep Slide 1 + 2 under `1:00`, finish RQ2 by `2:30`, and start Slide 7 no later than `4:10`.
- Every slide's script is also embedded inside the PowerPoint **Speaker Notes** of `report/final_presentation.pptx` so you can read directly in Presenter View while recording.
