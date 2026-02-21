import gc
import polars as pl
import pandas as pd
import numpy as np
import os
import json
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm

# --- 1. ПОДГОТОВКА И ОТБОР 600 ФИЧЕЙ ---
DATA_DIR = "data/"
os.makedirs("models_cv", exist_ok=True)
os.makedirs("oof_preds", exist_ok=True)

print("🔍 Отбор ТОП-600 признаков...")
report = pd.read_csv("feature_importance_report.csv", index_col=0)
# Берем более широкий набор универсалов и специалистов
universal = report.sort_values('mean_importance', ascending=False).head(400).index.tolist()
specialists = []
for t in [c for c in report.columns if c.startswith('target_')]:
    specialists.extend(report.sort_values(t, ascending=False).head(30).index.tolist())
big_smart_features = list(set(universal + specialists))[:600] # Ограничимся 600 для памяти

# --- 2. ЗАГРУЗКА ДАННЫХ ---
print(f"⏳ Загрузка 100% данных с {len(big_smart_features)} экстра-фичами...")
train_main = pl.read_parquet(f"{DATA_DIR}train_main_features.parquet")
train_target = pl.read_parquet(f"{DATA_DIR}train_target.parquet")
train_extra = pl.read_parquet(f"{DATA_DIR}train_extra_features.parquet", columns=["customer_id"] + big_smart_features)

train_df = train_main.join(train_extra, on="customer_id", how="inner").join(train_target, on="customer_id", how="inner").to_pandas()
del train_main, train_extra, train_target
gc.collect()

test_main = pl.read_parquet(f"{DATA_DIR}test_main_features.parquet")
test_extra = pl.read_parquet(f"{DATA_DIR}test_extra_features.parquet", columns=["customer_id"] + big_smart_features)
test_df = test_main.join(test_extra, on="customer_id", how="inner").to_pandas()
del test_main, test_extra
gc.collect()

feature_cols = [c for c in train_df.columns if c.startswith(("num_feature", "cat_feature"))]
target_cols = [c for c in train_df.columns if c.startswith("target_")]
cat_features = [c for c in feature_cols if c.startswith("cat_")]

for col in cat_features:
    train_df[col] = train_df[col].astype(str).fillna("NONE")
    test_df[col] = test_df[col].astype(str).fillna("NONE")

# --- 3. КРОСС-ВАЛИДАЦИЯ И ОБУЧЕНИЕ (LONG RUN) ---
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_predictions = pd.DataFrame(index=train_df.index)
final_test_predictions = pd.DataFrame(index=test_df.index)
final_test_predictions['customer_id'] = test_df['customer_id']

overall_scores = []

print(f"🚀 СТАРТ НОЧНОГО ПРОГОНА: 5-Fold CV на GPU")
print(f"Примерное время: 5-8 часов. Спокойной ночи! 🌙")

for target in tqdm(target_cols, desc="Targets"):
    y = train_df[target]
    oof_target = np.zeros(len(train_df))
    test_target_preds = np.zeros(len(test_df))
    
    # Внутренний цикл по фолдам
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
        X_train, y_train = train_df.loc[train_idx, feature_cols], y.iloc[train_idx]
        X_val, y_val = train_df.loc[val_idx, feature_cols], y.iloc[val_idx]
        
        ratio = (y_train == 0).sum() / (y_train == 1).sum() if y_train.sum() > 0 else 1
        
        model = CatBoostClassifier(
            iterations=2000, # Увеличили для максимального выучивания
            learning_rate=0.03, # Снизили для точности
            depth=6,
            scale_pos_weight=ratio,
            loss_function='Logloss',
            eval_metric='Logloss',
            random_seed=fold + 42, # Разный сид для каждого фолда
            verbose=0,
            task_type="GPU",
            devices='0',
            early_stopping_rounds=100
        )
        
        model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val))
        
        # OOF прогноз
        oof_target[val_idx] = model.predict_proba(X_val)[:, 1]
        # Прогноз на тест (усредняем)
        test_target_preds += model.predict_proba(test_df[feature_cols])[:, 1] / n_splits
        
        del model
        gc.collect()
        
    # Считаем скор для таргета по OOF
    score = roc_auc_score(y, oof_target)
    overall_scores.append(score)
    
    # Сохраняем результаты
    oof_predictions[target] = oof_target
    final_test_predictions[target.replace("target_", "predict_")] = test_target_preds

# --- 4. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ---
mean_cv_score = np.mean(overall_scores)
print(f"\n🏆 ФИНАЛЬНЫЙ CV ROC-AUC: {mean_cv_score:.6f}")

# Сохраняем OOF для стекинга
oof_predictions.to_parquet(f"oof_preds/oof_catboost_cv_{mean_cv_score:.4f}.parquet")

# Сохраняем сабмит
sub_name = f"SUBMISSION_FINAL_CV_{mean_cv_score:.4f}.parquet"
final_test_predictions.to_parquet(sub_name)

print(f"💾 Сабмит готов: {sub_name}")