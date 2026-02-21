import pandas as pd
from scipy.stats import rankdata

# 1. Твой лидер (0.8381)
df_best = pd.read_parquet("SUBMISSION_FINAL_CV_0.8310.parquet") 
# 2. Твой новый Титан (0.8362)
df_titan = pd.read_parquet("SUBMISSION_FINAL_TITAN_64BIT.parquet")

blend = df_best.copy()
cols = [c for c in df_best.columns if c != 'customer_id']

print("🚀 Смешиваем две мощные модели для прорыва...")

for col in cols:
    # Превращаем в ранги (это убирает проблему разного масштаба вероятностей)
    r_best = rankdata(df_best[col])
    r_titan = rankdata(df_titan[col])
    
    # Смешиваем ранги 50/50. Это даст синергию.
    blend[col] = (r_best * 0.5 + r_titan * 0.5) / len(df_best)

blend.to_parquet("SUBMISSION_ULTRA_ENSEMBLE.parquet")
print("✅ Готово! Загружай 'SUBMISSION_ULTRA_ENSEMBLE.parquet'.")