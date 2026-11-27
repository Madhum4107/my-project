# CPU & Memory Usage — Linear Trend Experiment

A beginner project that records CPU and memory usage for a short window and fits two simple linear-regression baselines to demonstrate model fitting and extrapolation.

> **Important limitation:** This is not a dependable forecasting or monitoring system. CPU and memory usage are time-varying, nonlinear and affected by workload. Extrapolating a short sample can be misleading. The chart is a learning visualization, not an operational alert.

## Run

Python 3.9+ recommended.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Install:
```bash
python -m pip install -r requirements.txt
```

Collect 20 readings (1 second apart), show 10 trend steps and plot:
```bash
python cpu_memory_predictor.py
```

Options:
```bash
python cpu_memory_predictor.py --samples 30 --interval 0.5 --horizon 5 --threshold 75 --no-plot
```

## Code structure

- `cpu_memory_predictor.py` — collection, validation, linear trend, threshold messages, plotting and CLI
- `test_cpu_memory_predictor.py` — deterministic tests using synthetic measurements

## Run tests

```bash
python -m unittest discover -v
```

## Method

Each metric is modeled independently against elapsed time. The predicted percentage is clipped to the valid 0–100 display range. Threshold messages describe the fitted trend, not a guaranteed future condition.

## Future improvements

- Compare against a persistence baseline and report MAE on a held-out time segment
- Use longer windows and rolling features
- Plot uncertainty and avoid extrapolating beyond a short horizon
- Separate collection from model evaluation

## Author

Madhumitha M · B.E. CSE (AI & ML), 2028
