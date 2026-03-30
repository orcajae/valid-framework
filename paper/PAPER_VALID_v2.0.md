# Beyond Accuracy: A Validation Framework for Machine Learning in Cryptocurrency Trading

---

*Working Paper — v2.0*
*Author: Jaewook Kim*
*Independent Researcher; B.S. Industrial Engineering, UNIST; Professional Engineer (Engineers Australia)*
*Date: April 2026*

---

## Highlights (≤85 characters each)

1. First financial ML validation checklist: 12-item VALID framework proposed
2. Bull bias: crypto ML models predict 90-97% long without class balancing
3. Statistical signals (PBO=0, p=0) do not guarantee economic profitability
4. Transaction costs consume 55-91% of ML alpha across five timeframes
5. Monte Carlo (n=100): AUC alone yields 29% false positive rate
6. First empirical confirmation of Witzany (2021) PBO critique via Monte Carlo

---

## Abstract

Machine learning strategies for cryptocurrency trading routinely report Sharpe ratios exceeding 2.0 in published research, yet practitioners consistently fail to replicate these results. We identify three systematic failure modes — directional prediction bias, the disconnect between statistical and economic significance, and transaction cost omission — through a comprehensive evaluation of 482 strategy variants across five timeframes (15-minute to weekly) for three cryptocurrency assets (BTC, ETH, SOL). To address these failures, we propose VALID (Validation Architecture for Learning-based Investment Decisions), a 12-item reporting and validation framework for financial ML research. VALID is the first domain-specific checklist for financial ML, analogous to TRIPOD+AI for clinical prediction models and REFORMS for general ML-based science.

We demonstrate VALID's necessity through three empirical contributions. First, we document that gradient-boosted tree models trained on cryptocurrency data without class balancing assign long signals to 90–97% of test observations across all three assets — a structural phenomenon we term "bull bias." Second, we show that standard validation tools (CPCV with PBO = 0.000, permutation test p = 0.000) certify non-overfitting and non-randomness, respectively, but cannot distinguish economically exploitable signals from statistically detectable noise — a disconnect confirmed by 100-iteration Monte Carlo analysis showing 29% false positive rates [95% CI: 21%, 39%] for AUC-based validation alone. Third, we quantify that transaction costs consume 55–91% of gross ML alpha, with ML strategies underperforming even the simplest rule-based alternatives (SMA crossover, RSI) at every cost level.

Additionally, we provide the first empirical confirmation of Witzany's (2021) theoretical critique of PBO through 100-iteration Monte Carlo simulation, deriving a null distribution of parameter-space variance and demonstrating that PBO = 0 in cryptocurrency ML reflects parameter-space flatness (Var(SR_IS) = 0.012) rather than signal robustness (null 95th percentile = 0.338).

Our proposed framework, validated against 482 strategy variants and benchmarked through Monte Carlo simulation, provides actionable reporting standards that would have flagged all identified failure modes prior to deployment. We release an open-source implementation for reproducibility.

**Keywords:** financial machine learning, cryptocurrency, validation framework, overfitting, CPCV, PBO, reporting standard, checklist

**JEL Classification:** G11, G14, G17, C45, C52

---

## 1. Introduction

The application of machine learning to cryptocurrency trading has grown explosively, producing a literature characterized by strikingly positive results. Cross-sectional factor models report weekly alphas of 3.87% (Fieberg et al., 2025). LSTM ensembles achieve annualized Sharpe ratios exceeding 3.0 (Xu et al., 2022). Tree-based models demonstrate AUC scores above 0.80 on directional prediction tasks (Li et al., 2024). Taken at face value, these findings suggest ML-driven crypto trading offers among the highest risk-adjusted returns in any asset class.

Yet the gap between published results and practitioner experience is vast. In traditional finance, this disconnect is well documented: Ioannidis (2005) demonstrated that most published research findings are false under conditions of low pre-study odds and high researcher flexibility. Harvey et al. (2016) showed that the majority of 316 published equity factors are likely false discoveries, proposing a t-ratio threshold of 3.0 rather than the conventional 2.0. McLean and Pontiff (2016) documented 26% out-of-sample decay and 58% post-publication decay in equity anomalies. Hou et al. (2020) replicated 452 anomalies and found 65% fail — 82% after multiple testing correction. The cryptocurrency ML literature, which combines the methodological challenges of financial prediction with the additional complexity of 24/7 markets, extreme volatility, and rapidly evolving market structure, has not yet undergone equivalent scrutiny.

Kapoor and Narayanan (2023) surveyed 329 ML papers across 17 scientific fields and identified data leakage in the majority, proposing a taxonomy of eight leakage types and a "model info sheet" reporting template. Their work, published in *Patterns* and subsequently extended to the REFORMS framework in *Science Advances* (2024), demonstrated that domain-specific validation standards can substantially reduce false positive rates. However, no analogous standard exists for financial ML. TRIPOD+AI (Collins et al., 2024) covers clinical prediction models. REFORMS addresses general ML-based science. MI-CLAIM (Norgeot et al., 2020) targets clinical AI. Not one published checklist addresses the unique challenges of ML-based trading strategy development: temporal data dependencies, transaction cost modeling, class imbalance in directional prediction, or the distinction between statistical and economic significance.

This gap has consequences. Without standardized validation, published crypto ML results vary wildly in methodological rigor. Some studies omit transaction costs entirely; others use unrealistically low cost assumptions. Few test for directional prediction bias. Even fewer apply advanced overfitting diagnostics such as Combinatorial Purged Cross-Validation (CPCV) or Probability of Backtest Overfitting (PBO), despite their demonstrated superiority over traditional walk-forward methods (Arian et al., 2024).

### 1.1 Contributions

This paper makes five contributions:

1. **The VALID framework.** We propose a 12-item validation and reporting checklist for financial ML research — the first domain-specific standard for this field. Each item is derived from an identified failure mode documented through systematic literature audit and empirical analysis. VALID is designed to complement existing standards (REFORMS, TRIPOD+AI) by addressing finance-specific pitfalls absent from general-purpose guidelines.

2. **Systematic literature audit.** We survey 80 published cryptocurrency ML trading papers (2018–2026), coding each for validation methodology, cost assumptions, class balancing, baseline comparison, and code availability. We quantify the prevalence of each failure mode the VALID framework is designed to detect.

3. **Three empirical failure modes at scale.** Using 482 strategy variants across five timeframes and three cryptocurrency assets (BTC, ETH, SOL), we document: (a) bull bias — ML models predicting 90–97% long across all assets without class balancing; (b) statistical-economic disconnect — validation tools certifying non-overfitting for signals that are economically marginal after costs; and (c) cost illusion — transaction costs consuming 55–91% of gross ML alpha.

4. **Monte Carlo false positive quantification.** We generate 100 synthetic price series with no embedded signal and run our full ML pipeline, demonstrating that standard AUC-based evaluation produces 29% false positives [21%, 39%] while the VALID-prescribed combination of CPCV + PBO reduces this to 0% [0%, 3.7%]. Ablation analysis identifies CPCV/PBO (Item V4) as the single most impactful validation component.

5. **Empirical validation of Witzany's PBO critique.** Through 100-iteration Monte Carlo simulation, we derive the first null distribution of parameter-space variance Var(SR_IS), demonstrating that PBO = 0.000 in cryptocurrency ML reflects parameter-space flatness (real Var = 0.012) rather than signal robustness (null 95th percentile = 0.338). This provides the first empirical confirmation of Witzany's (2021) theoretical critique and yields an actionable threshold for VALID Item V5.

### 1.2 Paper Organization

Section 2 reviews related work on financial ML methodology, crypto trading strategies, and existing reporting standards. Section 3 presents the systematic literature audit. Section 4 introduces the VALID framework. Sections 5 and 6 provide empirical validation through the three failure modes and Monte Carlo analysis, respectively. Section 7 discusses implications and limitations. Section 8 concludes.

---

## 2. Related Work

### 2.1 Machine Learning for Cryptocurrency Trading

The crypto ML literature has grown rapidly since 2019. Fang et al. (2022) surveyed 146 papers and identified technical analysis, fundamental analysis, and ML-based approaches as the three dominant paradigms, noting that approximately half of surveyed studies omit transaction costs. More recent work has expanded to cross-sectional factor models: Fieberg et al. (2025) developed the CTREND factor across 3,244 coins, while Cakici et al. (2024) applied 12 ML models to 37 cryptocurrency-specific factors. These studies report impressive risk-adjusted returns, typically in the range of Sharpe 1.5–3.0.

However, several concerns have emerged regarding the reliability of these results. Jaquart, Dann, and Weinhardt (2021) explicitly warned that imbalanced training sets may cause classifiers to predict the majority class regardless of input features in Bitcoin prediction tasks. Lahmiri and Bekiros (2024) found that naive models consistently outperform ML and deep learning in univariate crypto forecasting — a finding that received less attention than contemporaneous positive results. Grądzki et al. (2025) demonstrated that information-driven bars with triple barrier labeling improve upon standard approaches, but their cost assumptions (10 basis points) may understate realistic execution costs.

### 2.2 Backtest Overfitting and Validation Methodology

Bailey and López de Prado (2014, 2017) introduced PBO and DSR as tools for detecting backtest overfitting, demonstrating that the probability of selecting an overfit strategy increases rapidly with the number of configurations tested. Their CPCV framework, which creates multiple training-testing combinations respecting temporal ordering, has become the gold standard for financial strategy validation. Arian et al. (2024), published in Knowledge-Based Systems, compared CPCV against walk-forward methods in a synthetic environment, introducing Bagged CPCV and Adaptive CPCV variants. They found CPCV demonstrably superior in mitigating overfitting, as evidenced by lower PBO and superior DSR.

Witzany (2021) provided the only formal critique of PBO, demonstrating that CSCV/PBO exhibits negative bias when strategies have similar returns: "the best IS model tends to the worst OOS not because of the models but due to the design of the method itself." Bailey et al. (2017) themselves acknowledged that low PBO does not guarantee positive out-of-sample performance. These observations motivate our investigation of what PBO certifies versus what practitioners assume it certifies.

### 2.3 Statistical Versus Economic Significance

The distinction between statistical and economic significance has a deep literature in traditional finance. Harvey et al. (2016) argued that extensive data mining requires t-ratios above 3.0 for credibility. Novy-Marx and Velikov (2016) demonstrated that most equity anomalies with monthly turnover above 50% become unprofitable after realistic trading costs. Patton and Weller (2020) found momentum implementation costs of 7.2–7.6% annually eliminate most profits. Chen and Velikov (2023) showed that post-publication decay combined with trading costs destroys 93% of anomaly returns.

In cryptocurrency markets, the cost landscape is distinct. Makarov and Schoar (2020) documented significant cross-exchange price differences and arbitrage opportunities. Almeida and Gonçalves (2024) systematically reviewed crypto market microstructure, identifying bid-ask spreads of 2–50+ basis points depending on market conditions. No published study has systematically quantified the gross-to-net alpha gap across multiple timeframes for ML-based crypto strategies — a gap our work fills.

### 2.4 Reporting Standards for ML-Based Research

The need for domain-specific ML validation standards is increasingly recognized. TRIPOD (Collins et al., 2015) provides 22 items for clinical prediction model reporting and has accumulated over 4,000 citations. TRIPOD+AI (Collins et al., 2024) extends this to ML-based clinical models with 27 items. REFORMS (Kapoor et al., 2024), published in Science Advances, proposes 32 items across 8 modules for general ML-based science. MI-CLAIM (Norgeot et al., 2020), published in Nature Medicine, provides a 6-step checklist for clinical AI. Model Cards (Mitchell et al., 2019) propose documentation standards for trained ML models.

Critically, **no published reporting standard addresses financial ML or algorithmic trading**. The unique challenges of this domain — temporal data dependencies prohibiting random splits, the critical role of transaction costs, class imbalance in directional prediction, the distinction between statistical and economic significance, and regime-dependent model validity — are not covered by any existing framework. This gap motivates the VALID framework proposed in Section 4.

### 2.5 Cryptocurrency Momentum and Simple Benchmarks

Time-series momentum in Bitcoin was established by Liu and Tsyvinski (2021) and confirmed by subsequent studies. Yang et al. (2025) applied risk-managed momentum to crypto, reporting Sharpe improvements from 1.12 to 1.42. Grobys et al. (2025) documented that crypto momentum is subject to severe crashes mitigable through volatility management. These simple, transparent strategies serve as natural benchmarks for ML alternatives — yet many ML studies fail to include them as baselines, instead comparing only against buy-and-hold or other ML models.

---

## 3. Systematic Literature Audit

### 3.1 Search Strategy and Paper Selection

We searched Google Scholar, Scopus, and Web of Science for papers published between 2018 and 2026 using the query terms "cryptocurrency" AND ("machine learning" OR "deep learning") AND ("trading" OR "prediction" OR "forecasting"). We restricted to English-language papers published in peer-reviewed journals or established preprint servers (arXiv, SSRN). Priority was given to papers published in Finance Research Letters, Financial Innovation, Expert Systems with Applications, Knowledge-Based Systems, Journal of Finance and Data Science, International Review of Financial Analysis, and IEEE/ACM proceedings. After screening titles and abstracts for relevance to ML-based trading strategy development (excluding pure price forecasting without trading evaluation), we retained 80 papers for full-text analysis, of which 75 are empirical crypto ML studies (the remainder being surveys, synthetic environments, or equity-focused methodology papers included for comparative context).

### 3.2 Coding Methodology

Each paper was coded on seven dimensions corresponding to the VALID framework's core items:

| Dimension | Coding Question |
|-----------|----------------|
| D1: Cost modeling | Are transaction costs included? If so, what assumptions (basis points)? |
| D2: Class balance | Is directional prediction bias addressed? Method used? |
| D3: Temporal split | Is train/test splitting temporal or random? |
| D4: Validation method | Walk-forward, CPCV, k-fold, or simple holdout? |
| D5: Baseline comparison | Compared against buy-and-hold? Simple technical rules (SMA, RSI, MACD)? |
| D6: Economic evaluation | Is net (cost-adjusted) performance reported? |
| D7: Reproducibility | Is code or data publicly available? |

Coding was performed independently by the first author. For ambiguous cases, the most generous interpretation was applied (e.g., if a paper mentions costs in passing but does not clearly deduct them from reported returns, it was coded as "Partial" rather than "No").

### 3.3 Audit Results

Table 1 summarizes the audit findings across 75 empirical cryptocurrency ML papers, with 95% Wilson confidence intervals.

| Dimension | Failing | Rate | 95% CI | VALID Item |
|-----------|---------|------|--------|------------|
| D1: Transaction costs omitted | 40/75 | **53%** | [42%, 64%] | V7 |
| D2: Class balance not addressed | 54/75 | **72%** | [61%, 81%] | V1, V2 |
| D3: Random (non-temporal) split | 5/75 | 7% | [3%, 15%] | V3 |
| D4: Weak validation (holdout/random k-fold) | 15/75 | **20%** | [13%, 30%] | V4 |
| D4: CPCV used | 1/75 | **1%** | — | V4 |
| D5: Buy-and-hold only (no simple rule baselines) | 26/75 | **35%** | [25%, 46%] | V9 |
| D6: No net (cost-adjusted) performance reported | 40/75 | **53%** | [42%, 64%] | V7, V8 |
| D7: No code or data available | 63/75 | **84%** | [74%, 91%] | V12 |

Several patterns emerge. First, **cost omission remains widespread**: 53% of papers report no transaction costs [42%, 64%], consistent with Fang et al.'s (2022) survey estimate of approximately 50%. Among papers that include costs, assumptions range from 10 to 50 basis points per round trip — yet realistic costs on major exchanges (Binance, Bybit) for retail taker orders are 18–20 basis points round-trip including slippage, and this figure rises for less liquid altcoins or larger order sizes.

Second, **class imbalance is the most neglected dimension**: 72% of papers [61%, 81%] do not address class balance in directional prediction tasks. This is particularly concerning given that cryptocurrency returns are structurally positively skewed — a property that, as we demonstrate in Section 5.1, causes unbalanced classifiers to learn the base rate rather than conditional directional signals.

Third, **CPCV adoption is negligible**: only 1 of 75 papers (Arian et al., 2024, which specifically studies CPCV methodology) uses combinatorial purged cross-validation. The majority use walk-forward validation, which is adequate but does not provide PBO estimates or test for selection overfitting across configuration spaces. An additional 20% [13%, 30%] use only simple holdout or random k-fold splits on time-series data — methods that are known to produce optimistic evaluation due to temporal leakage.

Fourth, **baseline comparison is weak**: 35% [25%, 46%] compare only against buy-and-hold, without testing against simple technical rules (SMA crossover, RSI, MACD). This matters because, as we show in Section 5.3, even simple rule-based strategies (RSI > 50, SR = 0.804) outperform the best ML strategies in our evaluation.

Fifth, **reproducibility is poor**: 84% [74%, 91%] provide neither code nor data. Only 4 of 75 papers include a GitHub repository link.

### 3.4 Implications

The audit reveals that the VALID framework's 12 items address failure modes that are empirically prevalent. No audited paper satisfies all 12 VALID items. The highest-scoring papers in our audit satisfy 7–8 items (typically those published in finance journals with cost-inclusive backtesting and walk-forward validation, but lacking class balance analysis and CPCV). The median paper satisfies only 3–4 of 12 items. The audit data, including per-paper coding for all 80 papers, is available in the supplementary materials

---

## 4. The VALID Framework

### 4.1 Design Principles

The VALID (Validation Architecture for Learning-based Investment Decisions) framework is designed to address financial-ML-specific failure modes absent from existing standards. It follows three principles:

**Principle 1 — Finance-specific.** Each item addresses a pitfall unique to or especially severe in financial prediction, beyond what REFORMS or TRIPOD cover.

**Principle 2 — Evidence-based.** Each item is motivated by a quantified failure mode identified in the literature audit (Section 3) or empirical analysis (Sections 5–6).

**Principle 3 — Actionable.** Each item specifies what to report and how to test, not merely what to consider.

### 4.2 The 12 VALID Items

| # | Item | Rationale | Failure Mode Addressed |
|---|------|-----------|----------------------|
| V1 | **Report prediction class distribution** alongside accuracy/AUC. Report long%, short%, flat% on test set. | ML models on trending assets learn base rates. | Bull bias (Section 5.1) |
| V2 | **Test with and without class balancing.** Report both results. If unbalanced long% > 75%, flag as biased. | Class imbalance is structural in crypto (positive skew). | Bull bias |
| V3 | **Use temporal train-test splitting only.** No random k-fold on time series data. Report purge/embargo periods. | Information leakage across time invalidates OOS claims. | Temporal leakage |
| V4 | **Apply CPCV or equivalent** with PBO computation. Report PBO value and number of configurations tested. | Simple holdout is insufficient for high-dimensional search. | Backtest overfitting |
| V5 | **Report parameter-space variation.** Compute Var(SR_IS) across CPCV folds. If Var(SR_IS) falls below the 5th percentile of a Monte Carlo null distribution (~0.02 for crypto data; see Section 5.2.2), PBO should be flagged as reflecting parameter-space flatness. | PBO = 0 can arise from signal absence, not signal robustness (Witzany, 2021; empirically confirmed in Section 5.2.2). | PBO misinterpretation |
| V6 | **Include permutation tests** (≥100 shuffles). Observed AUC must exceed 95th percentile of null distribution. | Confirms signal existence independent of model selection. | Spurious patterns |
| V7 | **Report net performance** with explicit cost assumptions. Specify: exchange fees (maker/taker), slippage model, spread assumptions, funding rates if applicable. | Cost omission systematically inflates reported alpha. | Cost illusion (Section 5.3) |
| V8 | **Perform cost sensitivity analysis.** Report performance at 0bp, realistic (e.g., 18bp), and conservative (e.g., 50bp) round-trip costs. | Different cost assumptions can flip the sign of alpha. | Cost illusion |
| V9 | **Compare against simple baselines** — at minimum: buy-and-hold, a momentum strategy, and a technical indicator (SMA or RSI). Use identical cost assumptions. | Many ML studies compare only against buy-and-hold or other ML models, inflating relative performance. | Weak baseline selection |
| V10 | **Evaluate across at least one bear/correction period.** Report performance separately for bullish and bearish regimes. | Trend-following in bull markets produces positive results regardless of signal quality. | Regime overfitting |
| V11 | **Report trade frequency** and cost-per-unit-alpha. Compute: (annual cost drag) / (gross annual alpha). | High-frequency ML strategies face disproportionate cost headwinds. | Hidden turnover costs |
| V12 | **Provide code and data** for reproducibility. Minimum: preprocessing pipeline, model configuration, evaluation code. Recommended: Docker container, Zenodo DOI. | <10% of crypto ML papers provide code (audit finding). | Irreproducibility |

### 4.3 Relationship to Existing Standards

| Standard | Domain | Items | Finance-Specific | Cost Modeling | Class Balance |
|----------|--------|-------|------------------|---------------|---------------|
| TRIPOD+AI | Clinical ML | 27 | No | N/A | No |
| REFORMS | General ML | 32 | No | No | Partial |
| MI-CLAIM | Clinical AI | 6 | No | N/A | No |
| NeurIPS Checklist | ML research | 15 | No | No | No |
| **VALID** | **Financial ML** | **12** | **Yes** | **Yes (V7, V8, V11)** | **Yes (V1, V2)** |

VALID does not replace REFORMS — it supplements it with finance-specific items that general standards cannot address. Researchers should apply REFORMS for general ML rigor and VALID for financial deployment readiness.

### 4.4 Application Protocol

We recommend applying VALID in two stages:

**Stage 1 — Reporting (V1–V6, V9, V12):** These items address what to measure and report. They should be applied during research design and manuscript preparation.

**Stage 2 — Deployment Readiness (V7, V8, V10, V11):** These items address whether a strategy is viable for real-world deployment. They should be applied before any live trading or signal-selling operation.

A strategy that passes Stage 1 but fails Stage 2 has scientific value (the signal is real and not overfit) but limited practical value (the signal does not survive market frictions). This distinction — which we term the **statistical-economic disconnect** — is itself a contribution of the framework.

---

## 5. Empirical Validation: Three Failure Modes

Our 482 strategy variants comprise the following components, each evaluated through the full CPCV/PBO pipeline: (a) 75 Optuna-optimized hyperparameter configurations — 3 model families (CatBoost, LightGBM, Random Forest) × 5 timeframes (15-minute, 1-hour, 4-hour, daily, weekly), with the top 5 configurations per family per timeframe retained for CPCV evaluation; (b) 11 order flow feature variants tested on the 1-hour timeframe; (c) 20 cross-section momentum configurations across 20 cryptocurrency assets with varying lookback periods; (d) 8 volatility prediction overlay configurations; (e) 64 SHAP-derived rule combinations (4 thresholds × 3 macro indicators × grid); (f) 8 benchmark fortification variants (vol-scaling, circuit breaker, gold filter, funding rate filter, and their combinations); and (g) multi-asset portfolio combinations across BTC, ETH, and SOL. The total of 482 reflects unique strategy-timeframe-parameter configurations, each subjected to cost-inclusive evaluation at 18 basis points round-trip.

### 5.1 Failure Mode 1: Bull Bias

#### 5.1.1 Experimental Setup

We train gradient-boosted tree models (CatBoost, LightGBM, Random Forest) and deep learning models (2-layer LSTM, SimpleRNN) on hourly OHLCV data for three cryptocurrency assets: Bitcoin (BTC), Ethereum (ETH), and Solana (SOL). Each model uses 78 features spanning price, momentum, derivatives, on-chain, cross-asset, and microstructure categories, with Triple Barrier labeling and CUSUM event filtering following López de Prado (2018). We train each model twice: once with default class weights (unbalanced) and once with `class_weight='balanced'`.

#### 5.1.2 Results

| Asset | Unbalanced Long% | Balanced Long% | AUC (unbal) | AUC (bal) | Net SR |
|-------|------------------|----------------|-------------|-----------|--------|
| BTC (CatBoost) | 97.2% | 42.3% | 0.502 | 0.501 | −0.877 |
| BTC (LSTM) | 57.7% | 52.3% | 0.518 | 0.519 | −1.149 |
| BTC (SimpleRNN) | 63.3% | — | — | 0.520 | — |
| ETH (CatBoost) | 90.5% | 45.2% | 0.501 | 0.493 | −1.283 |
| SOL (CatBoost) | 97.0% | 32.2% | 0.533 | 0.511 | −0.206 |

Without class balancing, tree-based models exhibit extreme long bias (90–97%) while deep learning architectures show moderate bias (58–63%). AUC is indistinguishable from random (0.50–0.52) across all model families in both balanced and unbalanced settings. The severity of bull bias differs by architecture — tree-based models produce near-constant long predictions (90–97%), while LSTM and RNN exhibit moderate bias (58–63%) — but the economic outcome is identical: all architectures produce negative net Sharpe ratios after costs. The models have not learned directional prediction; they have learned varying approximations of the optimal constant prediction for positively skewed return distributions.

This finding extends the observation of Jaquart et al. (2021), who warned of majority-class prediction in Bitcoin ML, to a systematic, multi-asset, multi-model phenomenon. Bull bias is not a BTC-specific artifact — it is a structural consequence of cryptocurrency return distributions and affects all major crypto assets tested.

**VALID items that would have caught this:** V1 (prediction class distribution reporting) and V2 (mandatory class balancing test). Figure 1 visualizes the prediction distributions for balanced versus unbalanced models, illustrating the severity of bull bias and its impact on bear-market equity curves.

#### 5.1.3 Extended Multi-Asset, Multi-Timeframe Analysis

To confirm that bull bias is not an artifact of a single asset, timeframe, or model family, we extend the evaluation to 18 asset-timeframe-model combinations: 3 assets (BTC, ETH, SOL) × 2 timeframes (1-hour, daily) × 3 model families (CatBoost, LightGBM, Random Forest). Table 3 summarizes the results.

| Asset | Mean Unbal Long% | Mean Bal Long% | Mean AUC (bal) | Mean SR (net) |
|-------|------------------|----------------|----------------|---------------|
| BTC | 85% | 47% | 0.484 | −0.174 |
| ETH | 83% | 51% | 0.563 | +0.023 |
| SOL | 80% | 45% | 0.542 | +0.441 |

All assets exhibit unbalanced long ratios of 80–97% across all models and timeframes. The phenomenon is universal: no asset-model-timeframe combination produces unbalanced long ratios below 67%. Balanced AUC values cluster around 0.50 (BTC: 0.484, ETH: 0.563, SOL: 0.542), indicating that directional prediction is near-random for all three assets after bias correction.

Notably, all 1-hour models across all assets receive PBO = 1.000 (maximum overfitting), while daily models show higher AUC (0.54–0.65) but with PBO = 1.000 as well. This suggests that even when ML models achieve marginally above-random AUC, the signal does not survive rigorous overfitting testing.

Bull bias is not limited to hourly data or to BTC. Across five timeframes (15-minute to daily) on BTC alone, unbalanced models consistently produce long ratios exceeding 87%. The phenomenon is timeframe-invariant and asset-invariant, confirming its structural rather than incidental nature — a consequence of the positive skewness inherent in cryptocurrency return distributions.

### 5.2 Failure Mode 2: Statistical-Economic Disconnect

#### 5.2.1 The Disconnect

Our 1-hour CatBoost model (balanced) achieves PBO = 0.000 across 15 CPCV paths and permutation test p = 0.000 (observed AUC 0.570 vs. shuffled 95th percentile 0.516). By standard validation criteria, this model passes: it is not overfit (PBO = 0) and contains a real signal (permutation p < 0.05).

Yet its net Sharpe ratio is +0.135 — economically marginal. An ex-ante momentum benchmark achieves net SR of 0.917 with the same cost assumptions. The ML signal is **statistically real but economically unexploitable**.

This is not a failure of PBO or permutation testing. These tools correctly certify what they are designed to certify: PBO confirms the selection process did not overfit; the permutation test confirms the signal is non-random. The failure is the **assumption that statistical validity implies economic viability** — an assumption pervasive in the crypto ML literature but unsupported by theory or evidence.

The Var(SR_IS) across CPCV folds is 0.012. Against an ad hoc threshold of 0.01, this might appear to indicate a non-flat landscape. However, as we demonstrate in Section 5.2.2 through Monte Carlo simulation, this value falls dramatically below the null distribution (95th percentile = 0.338), revealing that the real parameter space is in fact substantially flatter than what random, structureless data produces. The IS-OOS Spearman correlation is −0.08 (p = 0.60), indicating that which configuration performs best in-sample does not predict which performs best out-of-sample, even though some signal transfers across all configurations.

This finding aligns with the broader literature on statistical-economic significance gaps. Harvey et al. (2016) argued that t-ratios above 2.0 are insufficient given extensive data mining. Novy-Marx and Velikov (2016) showed that most equity anomalies with high turnover fail after costs. Our contribution is to demonstrate this disconnect empirically in the crypto ML context, using the specific tools (CPCV, PBO, permutation) that the field relies upon.

**VALID items that would have caught this:** V5 (parameter-space variance), V7 (net performance reporting), V8 (cost sensitivity). Figure 2 illustrates the IS versus OOS Sharpe distribution across CPCV paths, showing the flat cluster pattern that characterizes the statistical-economic disconnect.

#### 5.2.2 Parameter-Space Flatness: Empirical Validation of Witzany (2021)

Witzany (2021) demonstrated theoretically that CSCV/PBO exhibits negative bias when strategies have similar returns, arguing that "the best IS model tends to the worst OOS not because of the models but due to the design of the method itself." Bailey et al. (2017, Section 5) acknowledged this limitation but provided no empirical test. Arian et al. (2024) compared CPCV variants in synthetic environments built from Heston and Merton jump-diffusion models but did not measure parameter-space variance distributions under null conditions. To our knowledge, no prior work has empirically quantified the null distribution of Var(SR_IS) — the variance in in-sample Sharpe ratios across tested configurations.

We address this gap using our 100-iteration Monte Carlo simulation (Section 6). For each null iteration, we compute Var(SR_IS) across the top CPCV-evaluated configurations, identical to the procedure applied to real data. The results are striking:

| Metric | Real Data | Null Distribution (100 iterations) |
|--------|-----------|-----------------------------------|
| Var(SR_IS) | 0.012 | Mean: 0.181, Median: 0.172, 95th pctl: 0.338 |

The real data's parameter-space variance (0.012) falls dramatically below the null distribution's 5th percentile. In other words, the variation in Sharpe ratios across ML configurations in real cryptocurrency data is *less* than what random, structureless data produces. This is the opposite of what would be expected if ML were capturing meaningful variation in strategy quality — genuinely different strategy configurations should produce more spread in performance, not less.

This result provides the first empirical confirmation of Witzany's (2021) theoretical critique. PBO = 0.000 in our BTC evaluation reflects a flat parameter landscape where no configuration is meaningfully superior to any other, not a robust signal that generalizes out-of-sample. The IS-OOS Spearman correlation of −0.08 (p = 0.60) reinforces this interpretation: knowing which configuration performed best in-sample provides no information about which will perform best out-of-sample.

The practical implication is immediate: researchers should report Var(SR_IS) alongside PBO (VALID Item V5) and interpret PBO = 0 with caution when Var(SR_IS) is low relative to a null distribution. We propose the following operational threshold: if Var(SR_IS) falls below the 5th percentile of a null distribution generated via Monte Carlo simulation with matched data characteristics, PBO should be flagged as potentially reflecting parameter-space flatness rather than genuine robustness. In our simulation, this threshold is approximately 0.02 — an empirically derived value that replaces the arbitrary 0.01 threshold suggested in earlier literature.

#### 5.2.3 Monte Carlo Confirmation

To quantify the false positive rate of standard validation approaches, we generate 100 synthetic BTC price series by bootstrapping daily returns (preserving marginal distribution but destroying temporal structure — see Section 6 for methodological details). For each synthetic series, we run the full ML pipeline: feature engineering, CatBoost with balanced weights, CPCV with PBO, and permutation testing.

| Validation Criterion | FPR (n=100) | 95% Wilson CI |
|---------------------|------------|---------------|
| AUC > 0.55 alone | 29.0% | [21.0%, 38.5%] |
| Permutation test passed | 22.0% | [15.0%, 31.1%] |
| PBO < 0.20 alone | 0.0% | [0.0%, 3.7%] |
| Net SR > 0 alone | 54.0% | [44.3%, 63.4%] |
| AUC > 0.55 AND PBO < 0.20 | 0.0% | [0.0%, 3.7%] |
| Full VALID (all criteria) | 0.0% | [0.0%, 3.7%] |

AUC-based evaluation alone would incorrectly validate nearly one-third of null signals [21%, 39%]. Even permutation testing, widely considered a robust control, produces 22% false positives [15%, 31%] — likely because the bootstrapped data preserves distributional properties that create spurious but statistically detectable patterns within individual folds. PBO eliminates all false positives in our simulation, with an upper confidence bound of 3.7%, confirming its value as the primary overfitting diagnostic. However, PBO does not prevent the adoption of economically unviable strategies (those with real but tiny signals). The VALID framework's combination of statistical validation (V4–V6) with economic evaluation (V7–V8, V11) addresses this limitation.

### 5.3 Failure Mode 3: Cost Illusion

#### 5.3.1 Cross-Timeframe Cost Analysis

We evaluate the best ML configuration at each timeframe alongside an ex-ante momentum benchmark, varying round-trip transaction costs from 0 to 50 basis points.

| Cost (RT bp) | Benchmark SR | 1h ML SR | 15m ML SR |
|-------------|-------------|---------|----------|
| 0 | 0.954 | 0.640 | 0.416 |
| 18 | 0.917 | 0.530 | 0.186 |
| 50 | 0.852 | 0.335 | −0.222 |

The critical finding: **ML strategies fail to beat the momentum benchmark at every cost level, including zero.** This means the cost illusion is not the only problem — the underlying ML signal is fundamentally weaker than simple momentum. Cost drag (55–91% across timeframes at 18bp) exacerbates the weakness but does not cause it.

We note that institutional traders with maker fee structures may face near-zero explicit costs. Our cost sensitivity analysis (Table 4) addresses this directly: even at 0bp round-trip costs, the best ML strategy (SR 0.640) fails to match the momentum benchmark (SR 0.954). The cost illusion therefore compounds a more fundamental problem — signal weakness — rather than creating it. This distinction is important: reducing transaction costs does not rescue ML alpha in our evaluation.

This result contrasts with published claims. Yang et al. (2025) reported vol-scaled momentum achieving SR 1.42 in crypto; when we apply their method to our single-asset BTC data with 18bp costs, the improvement reverses to a 14% SR degradation due to increased turnover (from 9.7 to 44.9 trades/year).

**VALID items that would have caught this:** V7 (net performance), V8 (cost sensitivity), V9 (simple baseline comparison), V11 (trade frequency reporting). Figure 3 visualizes the gross-to-net Sharpe degradation across timeframes, showing that cost drag increases monotonically with trading frequency.

#### 5.3.2 Comparison with Traditional Baselines

| Strategy | CAGR | MDD | SR (net) | Trades/yr |
|----------|------|-----|----------|-----------|
| Ex-ante momentum benchmark | 42.8% | −51.3% | 0.917 | 9.7 |
| RSI(14) > 50 | 38.4% | −54.1% | 0.804 | 44.0 |
| SMA 200 crossover | 33.1% | −64.6% | 0.723 | 7.6 |
| Buy & Hold | 30.6% | −76.6% | 0.618 | 0.1 |
| MACD crossover | 25.2% | −55.0% | 0.613 | 26.2 |
| 1h ML (balanced, CatBoost) | 19.4% | −62.6% | 0.530 | 30.8 |

The ML strategy ranks last among all tested approaches, including simple technical indicators (Figure 4 shows the monotonic relationship between strategy complexity and performance degradation). This finding does not imply ML is universally ineffective — it demonstrates that in the specific context of single-asset crypto directional prediction with realistic costs, the signal-to-noise ratio is insufficient for ML to add value over simple rules. Figure 5 presents regime-conditional performance, showing that the momentum benchmark's alpha concentrates in stressed market periods (SR 1.40 vs. 0.79 for buy-and-hold). Figure 6 shows the full distribution of net Sharpe ratios across all 482 variants, with the benchmark marked for reference — the vast majority of ML variants cluster below the benchmark.

### 5.4 Ablation Analysis: Which VALID Items Matter Most?

To quantify the relative importance of individual VALID items for false positive prevention, we systematically remove each item from the full VALID pipeline and measure the resulting false positive rate on our 100-iteration Monte Carlo simulation (Table 5).

| Configuration | Items Removed | FPR | 95% Wilson CI |
|--------------|--------------|-----|---------------|
| Full VALID | None | 0.0% | [0.0%, 3.7%] |
| No CPCV/PBO | −V4 | **29.0%** | [21.0%, 38.5%] |
| No class balancing | −V2 | 0.0% | [0.0%, 11.4%] |
| No cost adjustment | −V7 | 0.0% | [0.0%, 3.7%] |
| No permutation test | −V6 | 0.0% | [0.0%, 3.7%] |

The results reveal a clear hierarchy: **V4 (CPCV with PBO) is the single most important VALID item.** Its removal causes FPR to increase from 0% to 29% [21%, 39%] — a nearly thirty-fold increase. All other items, when individually removed in the presence of V4, show negligible standalone impact on FPR.

This does not mean V2, V7, or V6 are unimportant — they address failure modes (bull bias, cost illusion) that V4 cannot detect. A strategy may pass CPCV/PBO validation while still suffering from bull bias (Section 5.1) or cost-destroyed alpha (Section 5.3). Rather, V4 serves as the primary false-positive filter in the statistical domain, while V2, V7, and V9 address false-negative and economic viability concerns that operate in a fundamentally different dimension.

This finding has practical implications for resource allocation: researchers with limited computational budgets should prioritize CPCV/PBO (V4) above all other validation steps. If only one advanced validation technique can be applied, it should be CPCV with PBO — not permutation testing, not walk-forward, and not simple holdout.

---

## 6. Monte Carlo False Positive Analysis

The Monte Carlo simulation in Section 5.2.2 provides headline false positive rates. This section details the methodology and additional sensitivity analysis.

### 6.1 Synthetic Data Generation

We generate synthetic BTC price series by bootstrapping from the empirical distribution of daily log-returns (2018–2026, n = 2,974). Each synthetic series preserves the marginal distribution (mean, variance, skewness, kurtosis) while destroying temporal dependence through random sampling with replacement. Cumulative summation of sampled log-returns produces synthetic price paths of identical length to the original series. This approach ensures that any signal detected by the ML pipeline on synthetic data is, by construction, a false positive — the data contains realistic distributional properties but no exploitable temporal structure.

### 6.2 Pipeline Application

For each of 100 synthetic series, we apply the full ML pipeline: (a) compute all price-derived features (25 features from the 78-feature set, excluding macro and on-chain indicators which require external data sources); (b) generate Triple Barrier labels with CUSUM filtering using identical parameters to the empirical analysis (daily: pt = 2.0σ, sl = 2.5σ, max hold = 20 bars); (c) train CatBoost with balanced class weights (depth = 5, iterations = 100); (d) evaluate via CPCV (N = 6, k = 2, purge = 20 bars) with PBO computation; (e) record AUC, PBO, and hypothetical net Sharpe ratio.

### 6.3 Results

The mean AUC across 100 null iterations is 0.506 ± 0.070 (vs. theoretical 0.500), confirming that our feature set does not systematically extract spurious signal from random data. However, 29.0% [21.0%, 38.5%] of iterations produce AUC > 0.55 — the threshold commonly used to claim predictive power in published studies. Furthermore, 54.0% [44.3%, 63.4%] of iterations produce positive net Sharpe ratios purely by chance. This false positive rate drops to 0.0% [0.0%, 3.7%] when CPCV + PBO < 0.20 is required as a joint criterion, validating Items V4 and V6 of the VALID framework. The ablation analysis (Section 5.4) confirms that V4 (CPCV/PBO) alone accounts for the entire false positive prevention effect.

These rates have direct implications for interpreting the literature. If a typical crypto ML study uses AUC > 0.55 as its success criterion without CPCV/PBO validation, the probability of a false positive on structureless data is approximately one in three. Combined with the audit finding that only 3% of papers use CPCV (Section 3.3), this suggests a substantial fraction of published positive results may reflect statistical artifacts rather than genuine predictive signal.

---

## 7. Discussion

### 7.1 Implications for the Field

Our findings have four implications. First, the crypto ML literature likely contains a substantial false positive rate. If 53% of studies omit transaction costs (our audit) and 29% of AUC-based validations are false positives on null data (our Monte Carlo finding), the combined effect is that a significant fraction of published positive results may not survive proper validation.

Second, our Monte Carlo analysis provides the first empirical validation of Witzany's (2021) theoretical critique of PBO. We demonstrate that parameter-space flatness — not signal robustness — explains PBO = 0.000 in crypto ML applications, with real data Var(SR_IS) = 0.012 falling far below the null distribution's 95th percentile of 0.338. This finding has immediate practical implications: the field's most advanced overfitting diagnostic can give a "clean bill of health" to strategies that are, in fact, indistinguishable from noise. The empirically derived threshold of ~0.02 provides the first actionable criterion for flagging this failure mode.

Third, the statistical-economic disconnect is not a failure of existing tools but a gap in how they are applied. PBO and permutation tests work correctly — the problem is that researchers interpret statistical passing as sufficient evidence for practical viability, when these tests certify different properties entirely.

Fourth, the VALID framework addresses a genuine gap. No existing reporting standard covers the financial-ML-specific pitfalls documented in this paper. Adoption of VALID would provide a common vocabulary and minimum standard for evaluating crypto ML claims.

### 7.2 Limitations

Several limitations should be noted. Our empirical analysis covers three cryptocurrency assets; extending to equities, forex, or commodities would strengthen generalizability. The literature audit, while systematic, may not capture all relevant publications. The VALID framework has not been tested through formal expert consensus (e.g., Delphi process); we recommend this as future work. The Monte Carlo simulation uses 100 iterations; while this exceeds many published simulation studies, larger simulations may reveal additional edge cases in the tails of the null distribution.

The momentum benchmark operates at monthly frequency while ML strategies trade at hourly to daily frequency. This comparison measures risk-adjusted performance rather than strategy substitutability — the two approaches have different capacity profiles and microstructure exposures. For frequency-matched comparison, we note that RSI(14) > 50 (44 trades/year, SR 0.804) and MACD crossover (26 trades/year, SR 0.613) both outperform the best ML strategy (31 trades/year, SR 0.530) at comparable trading frequencies.

The momentum benchmark follows Antonacci's (2014) dual momentum framework with standard 12-month and 6-month lookback periods, evaluated monthly. We deliberately omit implementation details beyond the published framework (such as execution timing optimizations or supplementary risk management rules) as these do not affect the methodological conclusions. The core benchmark — four conditional momentum rules evaluated monthly — is fully specified and reproducible from Antonacci (2014).

### 7.3 Future Work

Three extensions merit investigation. First, applying the VALID framework retroactively to 10–20 highly cited crypto ML papers to assess what it would have caught. Second, extending the framework to cover reinforcement learning and generative model-based trading strategies. Third, developing an automated VALID compliance tool that can parse a research manuscript and flag potential violations.

---

## 8. Conclusion

We propose VALID, the first domain-specific validation and reporting framework for financial ML research. Through a systematic audit of 80 published papers, empirical analysis of 482 strategy variants across three crypto assets and five timeframes, 100-iteration Monte Carlo false positive analysis, and ablation testing, we demonstrate that the crypto ML literature suffers from three systematic failure modes: bull bias (80–97% long prediction without class balancing across all assets tested), statistical-economic disconnect (validation tools certifying signals that are economically marginal), and cost illusion (53–91% cost drag destroying ML alpha). We provide the first empirical confirmation of Witzany's (2021) theoretical critique of PBO, showing that parameter-space flatness in real cryptocurrency data (Var(SR_IS) = 0.012) falls far below the null distribution (95th percentile = 0.338), and derive an empirical threshold for VALID Item V5.

The VALID framework provides 12 actionable items that address each identified failure mode. Its adoption would establish minimum reporting standards for a field that currently lacks them, analogous to what TRIPOD achieved for clinical prediction and REFORMS for general ML-based science.

We close with the observation that motivates this work: in a field where positive results are rewarded and negative results are invisible, the most important validation is the discipline to accept that a statistically detectable signal is not the same as a tradable edge.

---

## References

Alessandretti, L., ElBahrawy, A., Aiello, L.M., & Baronchelli, A. (2018). Anticipating cryptocurrency prices using machine learning. Complexity, 2018, 8983590.

Almeida, J., & Gonçalves, T.C. (2024). Cryptocurrency market microstructure: A systematic literature review. Annals of Operations Research.

Antonacci, G. (2014). Dual Momentum Investing. McGraw-Hill.

Arian, H., Mobarekeh, D.N., & Seco, L. (2024). Backtest overfitting in the machine learning era: A comparison of out-of-sample testing methods in a synthetic controlled environment. Knowledge-Based Systems, 305, 112477.

Bailey, D.H., & López de Prado, M. (2014). The deflated Sharpe ratio. Journal of Portfolio Management, 40(5), 94–107.

Bailey, D.H., Borwein, J.M., López de Prado, M., & Zhu, Q.J. (2017). The probability of backtest overfitting. Journal of Computational Finance, 20(4), 39–69.

Blitz, D., Hanauer, M.X., Honarvar, I., Huisman, R., & van Vliet, P. (2023). Beyond Fama-French factors: Alpha from short-term signals. Financial Analysts Journal, 79(4), 96–117.

Borrageiro, G., Firoozye, N., & Barucca, P. (2022). Reinforcement learning for systematic FX trading. Expert Systems with Applications, 195, 116522.

Cakici, N., Shahzad, S.J.H., Będowska-Sójka, B., & Zaremba, A. (2024). Machine learning and the cross-section of cryptocurrency returns. International Review of Financial Analysis, 94, 103244.

Chen, A.Y., & Velikov, M. (2023). Zeroing in on the expected returns of anomalies. Journal of Financial and Quantitative Analysis, 58(3), 968–1004.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of KDD, 785–794.

Collins, G.S., et al. (2015). Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD). Annals of Internal Medicine, 162(1), 55–63.

Collins, G.S., et al. (2024). TRIPOD+AI statement. BMJ, 385, e078378.

Dorogush, A.V., Ershov, V., & Gulin, A. (2018). CatBoost: Gradient boosting with categorical features support. arXiv:1810.11363.

Fang, F., et al. (2022). Cryptocurrency trading: A comprehensive survey. Financial Innovation, 8, 13.

Fieberg, C., et al. (2025). Cryptocurrency factor momentum. JFQA, forthcoming.

Friedman, J.H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189–1232.

Grądzki, R., et al. (2025). Algorithmic crypto trading using information-driven bars. Financial Innovation, 11, 66.

Grobys, K., et al. (2025). Cryptocurrency momentum has (not) its moments. Financial Markets and Portfolio Management, forthcoming.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. Review of Financial Studies, 33(5), 2223–2273.

Hansen, P.R. (2005). A test for superior predictive ability. Journal of Business & Economic Statistics, 23(4), 365–380.

Harvey, C.R., Liu, Y., & Zhu, H. (2016). ...and the cross-section of expected returns. Review of Financial Studies, 29(1), 5–68.

Hou, K., Xue, C., & Zhang, L. (2020). Replicating anomalies. Review of Financial Studies, 33(5), 2019–2133.

Ioannidis, J.P.A. (2005). Why most published research findings are false. PLoS Medicine, 2(8), e124.

Jaquart, P., Dann, D., & Weinhardt, C. (2021). Short-term bitcoin market prediction via machine learning. Journal of Finance and Data Science, 7, 45–56.

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9), 100804.

Kapoor, S., et al. (2024). REFORMS: Consensus-based recommendations for machine-learning-based science. Science Advances, 10, eadk3452.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30.

Kim, A., Trimborn, S., & Härdle, W.K. (2021). VCRIX — A volatility index for crypto-currencies. International Review of Financial Analysis, 78, 101915.

Krauss, C., Do, X.A., & Huck, N. (2017). Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500. European Journal of Operational Research, 259(2), 689–702.

Lahmiri, S., & Bekiros, S. (2024). Complexity and predictability analysis of cryptocurrency time-series.

Leung, T., & Zhao, B. (2021). Cryptocurrency trading and exchanges. In Springer Handbook of Blockchain, 245–275.

Li, Y., et al. (2024). Cryptocurrency factors and machine learning. SSRN Working Paper.

Liu, Y., & Tsyvinski, A. (2021). Risks and returns of cryptocurrency. Review of Financial Studies, 34(6), 2689–2727.

Lo, A.W. (2002). The statistics of Sharpe ratios. Financial Analysts Journal, 58(4), 36–52.

López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.

Makarov, I., & Schoar, A. (2020). Trading and arbitrage in cryptocurrency markets. Journal of Financial Economics, 135(2), 293–319.

McLean, R.D., & Pontiff, J. (2016). Does academic research destroy stock return predictability? Journal of Finance, 71(1), 5–32.

McNally, S., Roche, J., & Caton, S. (2018). Predicting the price of Bitcoin using machine learning. 26th Euromicro Conference on PDP, 339–343.

Mitchell, M., et al. (2019). Model cards for model reporting. Proceedings of the Conference on Fairness, Accountability, and Transparency, 220–229.

Norgeot, B., et al. (2020). Minimum information about clinical artificial intelligence modeling: The MI-CLAIM checklist. Nature Medicine, 26, 1320–1324.

Novy-Marx, R., & Velikov, M. (2016). A taxonomy of anomalies and their trading costs. Review of Financial Studies, 29(1), 104–147.

Olorunnimbe, K., & Viktor, H. (2023). Deep learning in the stock market — a systematic survey of practice and research. Artificial Intelligence Review, 56, 5427–5501.

Patton, A.J., & Weller, B.M. (2020). What you see is not what you get. Journal of Financial Economics, 137(3), 515–549.

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. NeurIPS, 31.

Sebastião, H., & Godinho, P. (2021). Forecasting and trading cryptocurrencies with machine learning under changing market conditions. Financial Innovation, 7, 3.

Siami-Namini, S., Tavakoli, N., & Namin, A.S. (2019). The performance of LSTM and BiLSTM in forecasting time series. IEEE International Conference on Big Data.

Sun, X., Liu, M., & Sima, Z. (2020). A novel cryptocurrency price trend forecasting model based on LightGBM. Finance Research Letters, 32, 101084.

Vo, A., & Yuen, C. (2020). Towards multi-step cryptocurrency price prediction with deep learning. Expert Systems with Applications, 146, 113200.

White, H. (2000). A reality check for data snooping. Econometrica, 68(5), 1097–1126.

Witzany, J. (2021). A Bayesian approach to measurement of backtest overfitting. Risks, 9(1), 18.

Xu, W., et al. (2022). A machine learning approach for cryptocurrency trading. ScienceDirect.

Yang, L., et al. (2025). Cryptocurrency market risk-managed momentum strategies. Finance Research Letters, forthcoming.

Zhang, W., Li, P., Sha, D., Wang, Y., & Huang, S.H. (2025). Neural network-based algorithmic trading systems. arXiv:2508.02356.

Zhu, Y., Yang, Y., & Ren, Q. (2023). Machine learning in environmental research: Common pitfalls and best practices. Environmental Science & Technology, 57(46), 17671–17689.

---

## Data Availability

Bitcoin, Ethereum, and Solana OHLCV data were obtained from Binance and Bybit public APIs via the CCXT library. Macro indicators (S&P 500, VIX, DXY, Gold futures, US 10-year yield) were obtained from Yahoo Finance via the yfinance library. Funding rates were obtained from Binance and Bybit public APIs. The literature audit coding sheet for all 80 papers, Monte Carlo simulation results, and complete 482-variant performance data are provided in the supplementary materials. Pipeline source code, including the VALID checklist implementation, will be made available at a public GitHub repository upon publication.

---

*Appendix A: Complete 482-variant results (supplementary material)*
*Appendix B: VALID framework implementation code (GitHub repository)*
*Appendix C: Literature audit coding sheet and raw data*
*Appendix D: Monte Carlo simulation details*
