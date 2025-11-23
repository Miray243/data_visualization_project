import pandas as pd
import numpy as np

# ============================================================
# 1. VERİYİ OKU
# ============================================================
df_original = pd.read_csv("dataset.csv")

# ============================================================
# 2. GEREKSİZ KOLON SİL (Unnamed: 0)
# ============================================================
df = df_original.copy()
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# ============================================================
# 3. STRING KOLONLARDA TRIM UYGULA (boşlukları kaldır)
# ============================================================
object_cols = df.select_dtypes(include=["object"]).columns
df[object_cols] = df[object_cols].apply(lambda col: col.str.strip())

# ============================================================
# 4. EKSİK DEĞERLERİ DOLDUR
# ============================================================
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include=["object", "bool"]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ============================================================
# 5. AYKIRI DEĞERLERİ TEMİZLE (REVİZE EDİLDİ)
# ============================================================
# IQR yöntemini tüm kolonlara uygulamak yerine, sadece bariz hatalı olabilecek
# kolonlara manuel limit koymak daha iyi.

# Süresi 30 saniyeden kısa (30000 ms) veya 20 dakikadan uzun (1200000 ms) şarkıları at
df = df[(df['duration_ms'] > 30000) & (df['duration_ms'] < 1200000)]

# Tempo 0 olamaz, bunları at
df = df[df['tempo'] > 0]

# ============================================================
# 6. TEMİZLENMİŞ VERİYİ KAYDET
# ============================================================
df.to_csv("clean_dataset.csv", index=False)
print("Temiz veri 'clean_dataset.csv' olarak kaydedildi. Boyut:", df.shape)