import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

# تنظیمات هایپرپارامترها
config = {
    "hidden_layers": [512, 256, 128], "activation": "relu", "learning_rate": 0.0001,
    "epochs": 10000, "batch_size": 16384, "patience": 20, "test_size": 0.2, "val_size": 0.2, "random_state": 42
}

# تابع اضافه کردن ویژگی‌های اضافی
def add_features(x): 
    return np.hstack([x, (x[:, 5] / x[:, 4])[:, None], (x[:, 3] / x[:, 5])[:, None], (x[:, 2] / x[:, 3])[:, None]])

# بارگذاری و تقسیم داده‌ها
base_dir = os.path.dirname(os.path.abspath(__file__))
(x_data, y_data), _ = keras.datasets.california_housing.load_data()
x_train_full, x_test, y_train_full, y_test = train_test_split(x_data, y_data, test_size=config["test_size"], random_state=config["random_state"])
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=config["val_size"], random_state=config["random_state"])

# اضافه کردن ویژگی‌ها و نرمال‌سازی
x_train, x_val, x_test = map(add_features, [x_train, x_val, x_test])
mean, std = x_train.mean(0), x_train.std(0)
x_train, x_val, x_test = (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std

# ساخت و کامپایل مدل
model = keras.Sequential([keras.layers.Dense(u, activation=config["activation"]) for u in config["hidden_layers"]] + [keras.layers.Dense(1)])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]), loss="mse", metrics=["mae"])

# آموزش مدل
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=config["epochs"], batch_size=config["batch_size"],
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=config["patience"], restore_best_weights=True)])

# ارزیابی و ذخیره نتایج
y_pred = model.predict(x_test).flatten()
mae, mse, rmse, mape, r2 = mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred), \
                           np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_percentage_error(y_test, y_pred), r2_score(y_test, y_pred)
pd.DataFrame(history.history).to_csv(os.path.join(base_dir, "training_history.csv"), index=False)
with open(os.path.join(base_dir, "evaluation_results.txt"), "w") as f: 
    f.write(f"MAE: {mae}\nMSE: {mse}\nRMSE: {rmse}\nMAPE: {mape * 100:.2f}%\nR^2: {r2}\n")
model.save(os.path.join(base_dir, "trained_model.h5"))

# رسم و ذخیره نمودارها
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1); plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss (MSE)"); plt.legend()
plt.subplot(1, 3, 2); plt.plot(history.history['mae'], label='Train MAE'); plt.plot(history.history['val_mae'], label='Val MAE')
plt.title("Model MAE"); plt.legend()
plt.subplot(1, 3, 3); plt.plot(y_test, label='True'); plt.plot(y_pred, label='Pred'); plt.title("Predictions vs True"); plt.legend()
plt.savefig(os.path.join(base_dir, "training_history.png")); plt.show()

plt.figure(); plt.plot(abs(y_test - y_pred), color='red', label='Absolute Error'); plt.title("Absolute Error"); plt.legend()
plt.savefig(os.path.join(base_dir, "absolute_error.png")); plt.show()
