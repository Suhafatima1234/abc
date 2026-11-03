# ECG  Anomaly detection using Autoencoders 
# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 2. Load dataset (no column headers)
data = pd.read_csv("ecg.csv", header=None, sep=r"[\s,]+", engine="python")

# Separate features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

print("Dataset shape:", data.shape)
print("Labels:", np.unique(y))

# 3. Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Use only normal ECG samples for training
normal_label = 0
X_train = X[y == normal_label]

# 4. Define Encoder and Decoder
input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(encoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, output_layer)

# 5. Compile model
autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 6. Train model
history = autoencoder.fit(
    X_train, X_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1
)

# 7. Detect anomalies
reconstructed = autoencoder.predict(X)
error = np.mean(np.square(X - reconstructed), axis=1)

train_reconstructed = autoencoder.predict(X_train)
train_error = np.mean(np.square(X_train - train_reconstructed), axis=1)

threshold = np.percentile(train_error, 95)
predictions = (error > threshold).astype(int)

# 8. Evaluate model
print("Threshold:", threshold)
print("Confusion Matrix:\n", confusion_matrix(y, predictions))
print(classification_report(y, predictions))

# 9. Plot training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Autoencoder Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
