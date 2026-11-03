# 3C. Energy Consumption Anomaly Detection using LSTM Autoencoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

# 1. Load dataset
data = pd.read_csv("energy_consumption.csv")
data["Timestamp"] = pd.to_datetime(data["Timestamp"], format="%d-%m-%Y %H:%M")
data = data.sort_values("Timestamp")

labels = data["Anomaly_Label"].to_numpy()
X = data.drop(columns=["Timestamp", "Anomaly_Label"])
X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()

# 2. Split data
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_test = labels[split:]

# 3. Normalize data
scaler = StandardScaler()
train = scaler.fit_transform(X_train)
test = scaler.transform(X_test)

# 4. Create time windows
WINDOW = 10

def create_windows(data):
    return np.array([data[i:i+WINDOW]
                     for i in range(len(data)-WINDOW+1)])

X_train = create_windows(train)
X_test = create_windows(np.vstack([train[-(WINDOW-1):], test]))

# Train only on normal windows
train_labels = labels[:split]
label_windows = create_windows(train_labels.reshape(-1, 1))
X_train = X_train[np.all(label_windows[:, :, 0] == 0, axis=1)]

# Align test labels with test windows
test_labels = np.concatenate([labels[split-WINDOW+1:split], y_test])
y_test = np.any(create_windows(test_labels.reshape(-1, 1))[:, :, 0] == 1, axis=1).astype(int)

# 5. Build LSTM Autoencoder
inputs = Input(shape=(WINDOW, X_train.shape[2]))
encoded = LSTM(64)(inputs)
decoded = RepeatVector(WINDOW)(encoded)
decoded = LSTM(64, return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(X_train.shape[2]))(decoded)

model = Model(inputs, outputs)

# 6. Compile and train
model.compile(optimizer="adam", loss="mse")
history = model.fit(X_train, X_train, epochs=20,
                    batch_size=32, validation_split=0.1,
                    shuffle=False)

# 7. Calculate reconstruction error
train_error = np.mean((X_train - model.predict(X_train, verbose=0))**2, axis=(1, 2))
threshold = np.percentile(train_error, 95)

# 8. Detect anomalies
test_error = np.mean((X_test - model.predict(X_test, verbose=0))**2, axis=(1, 2))
y_pred = (test_error > threshold).astype(int)

print("Anomaly Threshold:", threshold)
print("Detected Anomalies:", np.sum(y_pred))

# 9. Evaluate model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 10. Plot training loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("LSTM Autoencoder Training")
plt.legend()
plt.show()