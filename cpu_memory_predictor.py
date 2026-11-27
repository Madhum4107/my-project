"""Collect a short system-usage sample and demonstrate a simple trend baseline.

This is a learning experiment, not a reliable resource forecasting/alerting tool.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable

import pandas as pd
import psutil
from sklearn.linear_model import LinearRegression

COLUMNS = ["time_seconds", "cpu_percent", "memory_percent"]


def collect_usage(samples: int = 20, interval: float = 1.0) -> pd.DataFrame:
    """Collect usage readings; the first CPU reading is discarded by psutil."""
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if interval <= 0:
        raise ValueError("interval must be greater than zero")
    rows = []
    for index in range(samples):
        cpu = psutil.cpu_percent(interval=interval)
        memory = psutil.virtual_memory().percent
        rows.append((index * interval, cpu, memory))
    return pd.DataFrame(rows, columns=COLUMNS)


def forecast_trend(data: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """Fit independent linear trends and clip percentage predictions to [0, 100]."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    missing = set(COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = data[COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 2:
        raise ValueError("At least two complete observations are required")
    if frame["time_seconds"].nunique() < 2:
        raise ValueError("time_seconds must contain at least two unique values")
    cadence = frame["time_seconds"].diff().dropna().median()
    cadence = float(cadence) if pd.notna(cadence) and cadence > 0 else 1.0
    future = pd.DataFrame({
        "time_seconds": [frame["time_seconds"].max() + cadence * (i + 1) for i in range(horizon)]
    })
    predictions = future.copy()
    for target in ["cpu_percent", "memory_percent"]:
        model = LinearRegression().fit(frame[["time_seconds"]], frame[target])
        predictions[target] = model.predict(future[["time_seconds"]]).clip(0, 100)
    return predictions


def predicted_alerts(predictions: pd.DataFrame, threshold: float = 80.0) -> list[str]:
    """Return human-readable threshold warnings; not guarantees of future load."""
    if not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")
    alerts = []
    for _, row in predictions.iterrows():
        t = row["time_seconds"]
        for metric in ["cpu_percent", "memory_percent"]:
            if row[metric] >= threshold:
                alerts.append(f"Trend baseline estimates high {metric.replace('_percent', '').upper()} at t={t:g}s")
    return alerts


def plot_usage(observed: pd.DataFrame, forecast: pd.DataFrame) -> None:
    """Plot collected measurements and the linear baseline."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for metric, label in [("cpu_percent", "CPU"), ("memory_percent", "Memory")]:
        ax.plot(observed["time_seconds"], observed[metric], marker=".", label=f"{label} observed")
        ax.plot(forecast["time_seconds"], forecast[metric], linestyle="--", label=f"{label} trend")
    ax.set(title="System Usage — Short Linear Trend Demo", xlabel="Elapsed time (seconds)", ylabel="Usage (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20, help="Number of measurements (default: 20)")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between readings")
    parser.add_argument("--horizon", type=int, default=10, help="Trend baseline steps to extrapolate")
    parser.add_argument("--threshold", type=float, default=80.0, help="Warning threshold, 0–100")
    parser.add_argument("--no-plot", action="store_true", help="Print output without opening a plot")
    args = parser.parse_args()

    observed = collect_usage(args.samples, args.interval)
    forecast = forecast_trend(observed, args.horizon)
    print("\nCollected readings:")
    print(observed.to_string(index=False))
    print("\nLinear trend baseline (not reliable forecasts):")
    print(forecast.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    alerts = predicted_alerts(forecast, args.threshold)
    print("\nThreshold warnings:")
    print("\n".join(alerts) if alerts else "No trend estimates crossed the configured threshold.")
    if not args.no_plot:
        plot_usage(observed, forecast)


if __name__ == "__main__":
    main()
