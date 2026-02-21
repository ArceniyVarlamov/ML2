import gc
import polars as pl
import pandas as pd
import numpy as np
import os
import json
import logging
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from tqdm import tqdm

# --- 1. НАСТРОЙКА ЛОГИРОВАНИЯ (UTF-8 для Windows) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("log_turbo.txt", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

DATA_DIR = "data/"
META_DIR = "metadata/"

def load_and_prepare():
    logging.info("START: Loading and Engineering Features...")
    
    # Загружаем список 600 лучших признаков
    with open(os.path.join(META_DIR, "smart_extra_features.json"), "r") as f:
        smart_features = json.load(f)
    
    # Ограничиваемся 600 признаками для стабильности памяти
    extra_cols = smart_features[:600] 
    
    # 1. Загружаем ТРЕЙН
    train_main = pl.read_parquet(os.path.join(DATA_DIR, "train_main_features.parquet"))
    train_extra = pl.read_parquet(os.path.join(DATA_DIR, "train_extra_features.parquet"), columns=["customer_id"] + extra_cols)
    train_target = pl.read_parquet(os.path.join(DATA_DIR, "train_target.parquet"))

    # Добавляем "Золотые" агрегаты
    train_extra = train_extra.with_columns([
        pl.mean_horizontal(extra_cols).cast(pl.Float32).alias("row_mean"),
        pl.sum_horizontal([(pl.col(c) == 0).cast(pl.Int32) for c in extra_cols]).alias("row_zeros"),
        pl.max_horizontal(extra_cols).cast(pl.Float32).alias("row_max")
    ])
    
    train_df = train_main.join(train_extra, on="customer_id", how="inner").join(train_target, on="customer_id", how="inner").to_pandas()
    
    # 2. Загружаем ТЕСТ
    test_main = pl.read_parquet(os.path.join(DATA_DIR, "test_main_features.parquet"))
    test_extra = pl.read_parquet(os.path.join(DATA_DIR, "test_extra_features.parquet"), columns=["customer_id"] + extra_cols)
    test_extra = test_extra.with_columns([
        pl.mean_horizontal(extra_cols).cast(pl.Float32).alias("row_mean"),
        pl.sum_horizontal([(pl.col(c) == 0).cast(pl.Int32) for c in extra_cols]).alias("row_zeros"),
        pl.max_horizontal(extra_cols).cast(pl.Float32).alias("row_max")
    ])
    test_df = test_main.join(test_extra, on="customer_id", how="inner").to_pandas()
    
    # Оптимизация памяти: Float64 -> Float32
    for df in [train_df, test_df]:
        floats = df.select_dtypes(include=['float64']).columns
        df[floats] = df[floats].astype(np.float32)

    # Категории в строки
    cat_features = [c for c in train_df.columns if c.startswith("cat_feature")]
    for col in cat_features:
        train_df[col] = train_df[col].astype(str).fillna("NONE")
        test_df[col] = test_df[col].astype(str).fillna("NONE")
        
    return train_df, test_df, cat_features

# Функция для параллельного обучения (Windows Friendly)
def train_single_product(target_name, train_df, test_df, feature_cols, cat_features):
    try:
        y = train_df[target_name].astype(np.int8)
        # 10 фолдов для максимального качества
        n_splits = 10
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        target_test_preds = np.zeros(len(test_df), dtype=np.float32)
        
        for fold, (idx_tr, idx_val) in enumerate(skf.split(train_df, y)):
            X_tr, y_tr = train_df.loc[idx_tr, feature_cols], y.iloc[idx_tr]
            X_val, y_val = train_df.loc[idx_val, feature_cols], y.iloc[idx_val]
            
            # Параметры "Extreme" для долгого и точного обучения
            model = CatBoostClassifier(
                iterations=1500,     # Глубокое обучение
                learning_rate=0.03,  # Медленный шаг для точности
                depth=6,
                l2_leaf_reg=5,       # Регуляризация
                scale_pos_weight=(y_tr == 0).sum() / y_tr.sum() if y_tr.sum() > 0 else 1,
                task_type="GPU",
                devices='0',         # Оба потока делят GPU
                verbose=0,
                early_stopping_rounds=100
            )
            
            model.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_val, y_val))
            target_test_preds += model.predict_proba(test_df[feature_cols])[:, 1] / n_splits
            
            del model, X_tr, X_val
            gc.collect()
            
        logging.info(f"✅ Product {target_name} finished.")
        return target_name.replace("target_", "predict_"), target_test_preds
    except Exception as e:
        logging.error(f"❌ Error in {target_name}: {e}")
        return target_name.replace("target_", "predict_"), None

# --- ТОЧКА ВХОДА ---
if __name__ == "__main__":
    print("--- TITAN TURBO SCRIPT STARTED ---")
    
    train_df, test_df, cat_features = load_and_prepare()
    gc.collect()

    feature_cols = [c for c in train_df.columns if c.startswith(("num_feature", "cat_feature", "row_"))]
    target_cols = [c for c in train_df.columns if c.startswith("target_")]

    logging.info(f"🚀 СТАРТ. Признаков: {len(feature_cols)}. Потоков: 2.")

    # n_jobs=2 ускоряет обучение в 2 раза за счет параллельности
    results = Parallel(n_jobs=1)(
        delayed(train_single_product)(t, train_df, test_df, feature_cols, cat_features) 
        for t in tqdm(target_cols, desc="Processing Products")
    )

    logging.info("📦 Сборка финального файла...")
    submission = pd.DataFrame({'customer_id': test_df['customer_id']})
    for col_name, preds in results:
        if preds is not None:
            submission[col_name] = preds
        else:
            submission[col_name] = 0.0 # Заглушка при ошибке

    sub_path = "SUBMISSION_TITAN_EXTREME_10FOLD.parquet"
    submission.to_parquet(sub_path)
    logging.info(f"🏆 ВСЁ ГОТОВО! Файл: {sub_path}")