# SQ4 — One-minute presentation script

Our fourth research question asks whether the CNN's visual signals transfer across market eras and which chart regions inform its predictions.

First, the CNN trained on 1993 to 1999 shows declining out-of-sample AUC, from 0.539 in the 2000s to 0.522 in the 2010s. We then retrain the identical architecture on 2006 to 2012 and compare both models on exactly 472,357 charts from 2013 to 2019. The modern model's AUC is 0.52218, versus 0.52168 for the historical model, but the paired monthly confidence interval crosses zero. Its cross-entropy and calibration are also worse. Newer training data therefore do not guarantee better transfer.

Second, Grad-CAM reveals a striking class difference across two thousand extreme predictions. Bearish forecasts allocate 20.3 percent of attribution to the volume region, compared with 6.9 percent for bullish forecasts. This is evidence about the CNN's decision patterns, not a causal market explanation.

Our takeaway is that temporal predictive performance and visual attribution must be audited separately.
