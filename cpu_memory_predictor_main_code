# cpu_memory_predictor.py

import psutil
import time
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data collection
data = []

print("Data collection starts... wait 20 seconds")

for i in range(20):
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent
    data.append([i, cpu, memory])
    print(f"Time {i}: CPU={cpu}%, Memory={memory}%")

# Data to DataFrame
df = pd.DataFrame(data, columns=["time", "cpu", "memory"])

# Train ML Model
X = df[["time"]]
cpu_model = LinearRegression()
mem_model = LinearRegression()
cpu_model.fit(X, df["cpu"])
mem_model.fit(X, df["memory"])

# Predict next 10 seconds
future_time = pd.DataFrame({"time": range(20, 30)})
cpu_pred = cpu_model.predict(future_time)
mem_pred = mem_model.predict(future_time)

# Show predictions
print("\nPredictions:")
for i in range(10):
    print(f"Time {20+i}: CPU={cpu_pred[i]:.2f}%, Memory={mem_pred[i]:.2f}%")

# Alert System
for i in range(10):
    if cpu_pred[i] > 80:
        print(f"⚠️ High CPU predicted at time {20+i}")
    if mem_pred[i] > 80:
        print(f"⚠️ High Memory predicted at time {20+i}")

# Plot Graph
plt.plot(df["time"], df["cpu"], label="CPU Actual")
plt.plot(df["time"], df["memory"], label="Memory Actual")
plt.plot(range(20, 30), cpu_pred, '--', label="CPU Predicted")
plt.plot(range(20, 30), mem_pred, '--', label="Memory Predicted")

plt.xlabel("Time")
plt.ylabel("Usage (%)")
plt.title("CPU & Memory Prediction")
plt.legend()
plt.show()
