import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from config import Config  # وارد کردن تنظیمات از فایل config.py

class Evaluator:
    """
    کلاس Evaluator مسئول ارزیابی مدل پس از آموزش و ذخیره نتایج ارزیابی است.

    این کلاس شامل متدی برای محاسبه معیارهای ارزیابی مختلف مدل و ذخیره آن‌ها به صورت یک فایل متنی است.
    """

    @staticmethod
    def evaluate_model(y_true, y_pred, config):
        """
        محاسبه و ذخیره معیارهای ارزیابی مدل.

        این متد معیارهای MAE، MAPE، RMSE و \( R^2 \) را برای مدل محاسبه می‌کند و آن‌ها را در یک فایل متنی ذخیره می‌نماید. 
        مسیر ذخیره‌سازی فایل نتایج ارزیابی از کلاس تنظیمات (Config) دریافت می‌شود.

        پارامترها:
        - y_true: مقادیر واقعی برچسب‌ها (آرایه Numpy)
        - y_pred: مقادیر پیش‌بینی شده توسط مدل (آرایه Numpy)
        - config: نمونه‌ای از کلاس تنظیمات (Config) که شامل مسیر ذخیره‌سازی فایل نتایج ارزیابی است
        """
        
        # محاسبه معیارهای ارزیابی
        mae = np.mean(np.abs(y_true - y_pred))  # میانگین خطای مطلق
        """
        MAE یا میانگین خطای مطلق، میانگین مقدار اختلافات مطلق بین مقادیر واقعی و پیش‌بینی شده را نشان می‌دهد.
        """

        mape = mean_absolute_percentage_error(y_true, y_pred)  # میانگین درصد خطای مطلق
        """
        MAPE یا میانگین درصد خطای مطلق، درصد اختلافات مطلق بین مقادیر واقعی و پیش‌بینی شده را نشان می‌دهد.
        """

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))  # جذر میانگین مربعات خطا
        """
        RMSE یا جذر میانگین مربعات خطا، نشان‌دهنده میزان اختلاف بین مقادیر واقعی و پیش‌بینی شده است و جذر MSE می‌باشد.
        """

        r2 = r2_score(y_true, y_pred)  # ضریب تعیین
        """
        R^2 یا ضریب تعیین، نشان‌دهنده میزان دقت مدل در پیش‌بینی داده‌ها است و بین 0 تا 1 می‌باشد.
        """

        # ذخیره نتایج ارزیابی به فایل متنی
        with open(config.evaluation_results_path, "w") as file:
            file.write("Evaluation Results:\n")
            file.write(f"Mean Absolute Error (MAE): {mae}\n")
            file.write(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%\n")
            file.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
            file.write(f"R-squared (R^2): {r2}\n")

        print(f"Evaluation results saved at {config.evaluation_results_path}")
