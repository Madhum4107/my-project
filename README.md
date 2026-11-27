# CPU & Memory Usage Predictor

A Python learning project that collects system resource usage, fits simple linear-regression models to the captured sample, and visualizes short-horizon CPU and memory trends.

## Features

- Capture CPU utilization and memory utilization with `psutil`
- Collect a short time-series sample
- Fit separate Linear Regression models to CPU and memory readings
- Predict the next 10 time steps (experimental)
- Print threshold-based high-usage warnings
- Plot observed and predicted values

## Workflow

```text
System metrics → Sample collection → DataFrame
        → Linear Regression → Forecast → Plot & threshold warnings
```

## Tech stack

Python · psutil · pandas · scikit-learn · matplotlib

## Run locally

```bash
git clone https://github.com/Madhum4107/my-project.git
cd my-project
python -m venv .venv
```

Activate the environment:

**Windows**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python cpu_memory_predictor.py
```

The script collects metrics for approximately 20 seconds before fitting its models.

## Notes & limitations

- This is an educational forecasting experiment, not a production system monitor.
- The model is trained on a very short sample and uses time as its only feature; predictions may be unstable and should not be interpreted as reliable forecasts.
- Threshold alerts are simple rules, not anomaly detection.
- The script opens a Matplotlib plot after prediction.

## Project report

[Read the project report (PDF)](The-Intelligent-System-CPU-and-Memory-Analysis-with-ML-Prediction.pdf)

## Author

**Madhumitha M** · [GitHub](https://github.com/Madhum4107) · [LinkedIn](https://www.linkedin.com/in/madhumitha4107/)
