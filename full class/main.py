# main.py
from config import Config
from data_loader import DataLoader
from model_builder import ModelBuilder
from trainer import Trainer
from evaluator import Evaluator

def main():
    """
    تابع اصلی برای اجرای فرآیند کامل آموزش و ارزیابی مدل شبکه عصبی.

    این تابع مراحل زیر را انجام می‌دهد:
    1. بارگذاری و پیش‌پردازش داده‌ها
    2. ساخت مدل با تنظیمات مشخص‌شده
    3. آموزش مدل و ذخیره تاریخچه آموزش
    4. ارزیابی مدل و ذخیره نتایج ارزیابی
    
    """
    # بارگذاری تنظیمات از کلاس Config
    config = Config()

    # مرحله 1: بارگذاری و آماده‌سازی داده‌ها
    data_loader = DataLoader()
    (x_train, y_train), (x_validate, y_validate) = data_loader.load_data()
    
    # اضافه کردن ویژگی‌های اضافی
    x_train = data_loader.add_features(x_train)
    x_validate = data_loader.add_features(x_validate)
    
    # نرمال‌سازی داده‌ها
    x_train = data_loader.normalize_data(x_train, training=True)
    x_validate = data_loader.normalize_data(x_validate, training=False)

    # مرحله 2: ساخت مدل
    model_builder = ModelBuilder(
        hidden_layers=config.hidden_layers,
        activation=config.activation,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate
    )
    model = model_builder.build_model(input_shape=(x_train.shape[1],))

    # مرحله 3: آموزش مدل
    trainer = Trainer(model, config)
    trainer.train(x_train, y_train, x_validate, y_validate)
    trainer.save_training_history()
    trainer.plot_training_history()

    # مرحله 4: ارزیابی مدل
    y_pred = model.predict(x_validate).flatten()
    Evaluator.evaluate_model(y_validate, y_pred, config)

    # مرحله 5: ذخیره مدل
    model.save(config.model_save_path)
    print(f"Model saved at {config.model_save_path}")


if __name__ == "__main__":

    main()
