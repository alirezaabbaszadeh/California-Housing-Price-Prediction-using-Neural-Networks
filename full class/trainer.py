import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from config import Config  # وارد کردن تنظیمات از فایل config.py



class Trainer:
    """
    کلاس Trainer مسئول مدیریت فرآیند آموزش مدل شبکه عصبی است.

    این کلاس شامل متدهایی برای آموزش مدل، ذخیره تاریخچه آموزش به فرمت CSV و رسم و ذخیره نمودار تاریخچه آموزش می‌باشد.
    """

    def __init__(self, model, config):
        """
        مقداردهی اولیه کلاس Trainer با استفاده از مدل و تنظیمات.

        پارامترها:
        - model: مدل شبکه عصبی که باید آموزش داده شود.
        - config: نمونه‌ای از کلاس تنظیمات (Config) که شامل پارامترهای آموزشی از جمله تعداد دوره‌ها، اندازه بچ و صبر برای توقف زودهنگام می‌باشد.
        """
        self.model = model  # مدل شبکه عصبی
        self.config = config  # تنظیمات آموزش شامل هایپرپارامترها و مسیرهای ذخیره‌سازی
        self.history = None  # تاریخچه آموزش مدل (نگهدارنده اطلاعات آموزش در هر دوره)

    def train(self, x_train, y_train, x_validate, y_validate):
        """
        آموزش مدل با داده‌های آموزشی و اعتبارسنجی و ذخیره تاریخچه آموزش.

        این متد از توقف زودهنگام (Early Stopping) برای جلوگیری از بیش‌برازش مدل استفاده می‌کند.
        تنظیمات مربوط به تعداد دوره‌ها، اندازه بچ و صبر توقف زودهنگام از کلاس Config دریافت می‌شود.

        پارامترها:
        - x_train: داده‌های آموزشی
        - y_train: برچسب‌های داده‌های آموزشی
        - x_validate: داده‌های اعتبارسنجی
        - y_validate: برچسب‌های داده‌های اعتبارسنجی
        """
        # تنظیم توقف زودهنگام با تعداد صبر مشخص
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.patience,
            restore_best_weights=True
        )

        # آموزش مدل و ذخیره تاریخچه
        self.history = self.model.fit(
            x_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(x_validate, y_validate),
            callbacks=[early_stopping],
            verbose=1
        )

    def save_training_history(self):
        """
        ذخیره تاریخچه آموزش به صورت فایل CSV.

        این متد تاریخچه آموزش شامل خطا و دقت مدل در هر دوره آموزشی را به فرمت CSV در مسیر مشخص‌شده در کلاس Config ذخیره می‌کند.
        """
        if self.history:
            history_df = pd.DataFrame(self.history.history)
            history_df.to_csv(self.config.history_csv_path, index=False)
        else:
            print("Error: Training history is empty. Please train the model first.")

    def plot_training_history(self):
        """
        رسم و ذخیره نمودار تاریخچه آموزش به صورت تصویر PNG.

        این متد نمودار تغییرات خطا و دقت مدل را در هر دوره آموزشی و اعتبارسنجی رسم کرده و در مسیر مشخص‌شده در کلاس Config ذخیره می‌کند.
        """
        if self.history:
            plt.figure(figsize=(14, 5))

            # رسم نمودار خطا (MSE) برای آموزش و اعتبارسنجی
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss (MSE)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.legend()

            # رسم نمودار MAE برای آموزش و اعتبارسنجی
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['mae'], label='Training MAE')
            plt.plot(self.history.history['val_mae'], label='Validation MAE')
            plt.title('Model MAE')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Absolute Error (MAE)')
            plt.legend()

            # ذخیره نمودار به صورت فایل تصویر
            plt.savefig(self.config.plot_path)
            plt.show()
        else:
            print("Error: Training history is empty. Please train the model first.")
