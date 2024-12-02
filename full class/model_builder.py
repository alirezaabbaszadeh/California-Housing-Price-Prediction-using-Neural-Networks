from tensorflow import keras
from tensorflow.keras import layers

class ModelBuilder:
    """
    کلاس ModelBuilder مسئول ساخت و کامپایل مدل شبکه عصبی است.

    این کلاس به کاربران اجازه می‌دهد که هایپرپارامترهای مدل را از طریق پارامترهای تنظیم شده تغییر دهند. 
    با استفاده از این کلاس، مدل می‌تواند با معماری‌های مختلف و بهینه‌سازهای متفاوت ساخته شود.
    """

    def __init__(self, hidden_layers=[128, 64, 32], activation="relu", optimizer="adam", learning_rate=0.001):
        """
        مقداردهی اولیه کلاس ModelBuilder با هایپرپارامترهای قابل تنظیم.

        پارامترها:
        - hidden_layers: لیستی از تعداد نورون‌ها در هر لایه مخفی مدل
        - activation: تابع فعال‌سازی برای لایه‌های مخفی
        - optimizer: بهینه‌ساز برای تنظیم وزن‌های مدل
        - learning_rate: نرخ یادگیری بهینه‌ساز
        """
        self.hidden_layers = hidden_layers  # لیستی از تعداد نورون‌ها در هر لایه مخفی
        """
        لیست تعداد نورون‌ها در هر لایه مخفی. برای مثال، [128, 64, 32] سه لایه با 128، 64 و 32 نورون ایجاد می‌کند.
        """

        self.activation = activation  # تابع فعال‌سازی لایه‌های مخفی
        """
        تابع فعال‌سازی که برای هر لایه مخفی استفاده می‌شود. تابع ReLU (Rectified Linear Unit) به طور پیش‌فرض تنظیم شده است.
        """

        self.optimizer = optimizer  # بهینه‌ساز مدل
        """
        نوع بهینه‌ساز که برای تنظیم وزن‌های مدل استفاده می‌شود. پیش‌فرض بهینه‌ساز Adam است.
        """

        self.learning_rate = learning_rate  # نرخ یادگیری
        """
        نرخ یادگیری برای بهینه‌ساز که تعیین می‌کند مدل با چه سرعتی باید وزن‌های خود را تنظیم کند.
        مقدار پیش‌فرض 0.001 است.
        """

    def build_model(self, input_shape):
        """
        ساخت و بازگشت مدل شبکه عصبی با معماری تنظیم شده.

        این متد یک مدل Sequential ایجاد می‌کند که شامل لایه‌های مخفی با تعداد نورون‌ها و تابع فعال‌سازی مشخص شده در `hidden_layers` و `activation` است. همچنین، یک لایه خروجی با یک نورون اضافه می‌شود که خروجی نهایی را تولید می‌کند.

        پارامتر:
        - input_shape: شکل ورودی داده‌ها، به صورت یک تاپل که نشان‌دهنده ابعاد ورودی است.

        خروجی:
        - model: مدل کامپایل‌شده شبکه عصبی
        """
        model = keras.Sequential()  # ایجاد مدل Sequential
        model.add(layers.Input(shape=input_shape))  # لایه ورودی

        # اضافه کردن لایه‌های مخفی بر اساس تعداد نورون‌ها و تابع فعال‌سازی مشخص شده
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation=self.activation))
        
        model.add(layers.Dense(1))  # اضافه کردن لایه خروجی با یک نورون

        # کامپایل مدل با بهینه‌ساز و نرخ یادگیری مشخص
        optimizer = self.get_optimizer()
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        
        return model

    def get_optimizer(self):
        """
        بازگشت بهینه‌ساز با نرخ یادگیری مشخص.

        این متد بهینه‌سازی که در `optimizer` مشخص شده را با استفاده از نرخ یادگیری `learning_rate` ایجاد می‌کند.
        به طور پیش‌فرض، Adam و SGD پشتیبانی می‌شوند.

        خروجی:
        - optimizer: نمونه‌ای از بهینه‌ساز با نرخ یادگیری مشخص

        خطا:
        - اگر نوع بهینه‌ساز پشتیبانی نشود، یک خطای ValueError صادر می‌شود.
        """
        if self.optimizer == "adam":
            return keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == "sgd":
            return keras.optimizers.SGD(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Optimizer '{self.optimizer}' is not supported.")
