# Figure alt text

## Figure 1 — Primary five-path hard-CBM computation

Flow diagram. Document text enters five seed-specific paths (seeds 1, 42, 1337, 2024, 2025). In each path, a concept model maps the text to ten concept scores; shared thresholds τ1–τ10 turn the scores into a ten-bit hard vector; and the primary logistic aggregator g_hard maps that vector to a path score. A dashed line marks the hard bottleneck: only the ten bits cross it, and text, encoder states and continuous scores do not reach the aggregator. The five path scores are averaged after they are produced, and the mean is compared with the final threshold τy = 0.41 to give the predicted class.

## Figure 2 — Bottleneck-vector diagnostics and full-text controls

Horizontal dot chart of test macro-F1, axis ticks from 0.75 to 0.90. Bottleneck-vector diagnostics: predicted bits, archived CBM, 0.778; gold bits, linear rule fitted on training data, 0.768; gold bits, majority-profile rule fitted on training data, 0.771; gold bits, exact optimum fitted on the test set, 0.826, shown as a hollow diamond with a dotted vertical reference line and labelled as selected with access to test labels. Full-text controls: TF–IDF 0.869; PhoBERT 0.906. An arrow at the top marks the 8.0-point shortfall from the gold-bit optimum to PhoBERT.
