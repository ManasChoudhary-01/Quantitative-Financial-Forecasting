# Financial Time-Series Foundation Model Benchmark

A benchmarking framework for evaluating pretrained time-series foundation models on financial forecasting tasks, with an emphasis on multi-horizon and multi-target prediction.

## Overview

This project studies how general-purpose time-series foundation models perform on financial data. It brings together a collection of 11 models—including **TimesFM, Moirai, and Chronos**—and evaluates their forecasting performance using a consistent zero-shot benchmarking workflow.

The benchmark covers **2,135 evaluation windows drawn from S&P 500 data**. The goal is to compare model behaviour across forecasting horizons and targets, understand where different architectures perform well, and identify promising configurations for financial time-series research.

> **Scope note:** This repository describes a forecasting and model-benchmarking project. It does not claim live trading, production deployment, or demonstrated investment returns. Alpha generation and portfolio construction are possible extensions, not assumed benchmark outcomes.

## Project Highlights

- **11 time-series foundation models** evaluated, including TimesFM, Moirai, and Chronos.
- **2,135 S&P 500 evaluation windows** used in the benchmarking study.
- **Zero-shot evaluation** to study pretrained model performance without task-specific model training, subject to each model's supported inference interface.
- **Multi-horizon and multi-target forecasting** to compare predictions across different forecast lengths and targets.
- A consistent evaluation workflow to make model comparisons easier to reproduce and analyse.

## Models

The benchmark includes 11 foundation models. The currently highlighted examples are:

| Model | Description |
|---|---|
| [TimesFM](https://github.com/google-research/timesfm) | Google's pretrained time-series foundation model. |
| [Moirai](https://github.com/SalesforceAIResearch/uni2ts) | A universal time-series forecasting model from Salesforce AI Research. |
| [Chronos](https://github.com/amazon-science/chronos-forecasting) | A family of pretrained probabilistic time-series forecasting models from Amazon. |

The remaining models should be listed here once their exact names and versions are finalized. Model versions, inference settings, and any model-specific adaptations should be recorded in the benchmark configuration to keep comparisons reproducible.

## Benchmark Workflow

```text
Financial time-series data
          |
          v
Data preparation and window generation
          |
          v
Input / target construction
          |
          v
Foundation-model inference (zero-shot)
          |
          v
Multi-horizon, multi-target predictions
          |
          v
Forecast evaluation and model comparison
          |
          v
Results analysis
```

### 1. Data preparation

Prepare the financial time-series data, apply consistent preprocessing, and create evaluation windows. The benchmark uses 2,135 windows drawn from S&P 500 data.

### 2. Forecasting

Run the selected foundation models on the prepared inputs. The workflow supports comparisons across multiple forecast horizons and targets. Since model interfaces can differ, model-specific input handling should be documented alongside the shared evaluation settings.

### 3. Evaluation

Compare model forecasts against the corresponding ground-truth targets. Report the metrics actually calculated by the implementation, broken down by model, forecast horizon, and target where applicable.

### 4. Analysis

Inspect aggregate performance as well as horizon-wise and target-wise differences. The analysis is intended to reveal relative model strengths and weaknesses—not to imply that lower forecasting error automatically translates into profitable trading strategies.

## Evaluation Considerations

For a fair comparison, record and keep consistent wherever possible:

- Evaluation-window definitions and chronological ordering.
- Input context length and forecast horizons.
- Target definitions and scaling / inverse-scaling procedures.
- Model checkpoint or package version and inference settings.
- Missing-data handling and any model-specific preprocessing.
- The exact metric definitions and aggregation method.

A results table can be added once benchmark outputs are available:

| Model | Target | Horizon | Metric 1 | Metric 2 |
|---|---|---:|---:|---:|
| TimesFM | — | — | — | — |
| Moirai | — | — | — | — |
| Chronos | — | — | — | — |

Replace the placeholders with measured results from the actual evaluation runs. Avoid comparing scores produced with different target scaling or evaluation-window definitions.

## Reproducibility

To make the benchmark reproducible, the repository should track:

1. Dataset source, retrieval date, and any applicable usage restrictions.
2. Data preprocessing and window-generation configuration.
3. Model names, versions, checkpoints, and inference parameters.
4. Evaluation scripts, metric definitions, and random seeds where relevant.
5. Hardware and software environment details for each run.

Dataset access instructions and exact run commands should be added here to match the scripts and configuration files included in this repository. No dataset is bundled or redistributed by this README.

## Potential Extensions

The current focus is forecasting and benchmarking. Possible next steps include:

- Convert forecast outputs into candidate features for statistical alpha research.
- Test whether forecast-derived features add predictive value beyond conventional financial features.
- Evaluate signal stability, information coefficient, and signal decay using strictly out-of-sample data.
- Explore signal combination and transaction-cost-aware portfolio backtesting.

These extensions require separate experiments and should only be described as completed once implemented and evaluated.

## References

- [TimesFM — Google Research](https://github.com/google-research/timesfm)
- [Moirai / Uni2TS — Salesforce AI Research](https://github.com/SalesforceAIResearch/uni2ts)
- [Chronos — Amazon Science](https://github.com/amazon-science/chronos-forecasting)

## Disclaimer

This project is for research and educational purposes only. Forecasting results are not investment advice and should not be interpreted as evidence of future financial performance.
