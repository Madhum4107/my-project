import unittest
import pandas as pd

from cpu_memory_predictor import forecast_trend, predicted_alerts


class TrendBaselineTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "time_seconds": [0, 1, 2],
            "cpu_percent": [10, 20, 30],
            "memory_percent": [40, 41, 42],
        })

    def test_forecast_returns_horizon_and_columns(self):
        result = forecast_trend(self.frame(), 3)
        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns), ["time_seconds", "cpu_percent", "memory_percent"])
        for actual, expected in zip(result["cpu_percent"], [40, 50, 60]):
            self.assertAlmostEqual(actual, expected)

    def test_future_timestamps_use_observed_cadence(self):
        data = self.frame()
        data["time_seconds"] = [0, 0.5, 1.0]
        result = forecast_trend(data, 2)
        self.assertAlmostEqual(result["time_seconds"].iloc[0], 1.5)
        self.assertAlmostEqual(result["time_seconds"].iloc[1], 2.0)

    def test_clips_percentages_to_valid_range(self):
        data = self.frame()
        data["cpu_percent"] = [90, 100, 110]
        result = forecast_trend(data, 4)
        self.assertTrue(result["cpu_percent"].between(0, 100).all())

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            forecast_trend(self.frame(), 0)
        with self.assertRaises(ValueError):
            forecast_trend(self.frame().iloc[:1], 2)

    def test_alert_threshold(self):
        result = pd.DataFrame({
            "time_seconds": [1, 2],
            "cpu_percent": [79, 80],
            "memory_percent": [20, 85],
        })
        alerts = predicted_alerts(result, 80)
        self.assertEqual(len(alerts), 2)

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            predicted_alerts(self.frame(), -1)


if __name__ == "__main__":
    unittest.main()
