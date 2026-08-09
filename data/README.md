# Data Directory

Raw market data is intentionally **not** tracked in git (see root `.gitignore`).

## Layout

```
data/
├── raw/          # Per-ticker OHLCV parquet files (gitignored, regenerated)
├── processed/    # Engineered features (gitignored, regenerated)
└── external/     # Third-party reference data (gitignored)
```

## Regeneration

The full raw dataset can be reproduced by running the data pipeline:

```bash
python run_pipeline.py --stage data
```

This pulls 10 years of daily OHLCV data for the S&P 100 tickers defined in
`config.yaml` (`data.tickers`) and writes one parquet file per ticker to
`data/raw/`.
