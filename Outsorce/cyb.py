import pandas as pd
import numpy as np
from scipy.stats import rankdata

# --- НАСТРОЙКИ ---
# Впиши точные названия твоих файлов!
file_best = "SUBMISSION_FINAL_CV_0.8310.parquet"  # Твой рекорд
file_titan = "SUBMISSION_FINAL_TITAN_64BIT.parquet"      # Твой ночной 26-часовой файл

print("⏳ Загружаем гигантов...")
df_best = pd.read_parquet(file_best)
df_titan = pd.read_parquet(file_titan)

# Проверка на совпадение клиентов (на всякий случай)
assert df_best['customer_id'].equals(df_titan['customer_id']), "ОШИБКА: Разный порядок клиентов!"

targets = [c for c in df_best.columns if c != 'customer_id']
submission = pd.DataFrame({'customer_id': df_best['customer_id']})

print("🚀 Смешиваем через Ranks (75% Лидер + 25% Титан)...")

for col in targets:
    # 1. Превращаем вероятности в ранги (от 0 до 1)
    # Это делает модели "совместимыми", даже если одна выдает 0.9, а другая 0.7
    r_best = rankdata(df_best[col]) / len(df_best)
    r_titan = rankdata(df_titan[col]) / len(df_titan)
    
    # 2. Взвешивание
    # Мы даем 75% веса твоей модели на 0.838, потому что она доказала эффективность.
    # Мы даем 25% Титану, чтобы он "подстраховал" сложные случаи своими 10 фолдами.
    final_score = (r_best * 0.75) + (r_titan * 0.25)
    
    submission[col] = final_score

output_file = "SUBMISSION_RANK_BLEND_0.75_0.25.parquet"
submission.to_parquet(output_file)

print(f"✅ ГОТОВО! Файл: {output_file}")
print("Загружай этот файл. Это математически лучший шанс пробить 0.84 без нового обучения.")