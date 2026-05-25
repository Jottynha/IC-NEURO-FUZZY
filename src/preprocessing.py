from ucimlrepo import fetch_ucirepo
import pandas as pd

# =========================================
# 1) ADULT
# =========================================
adult = fetch_ucirepo(id=2)
X_adult = adult.data.features
y_adult = adult.data.targets
print("\n===== ADULT =====")
print(X_adult.head())
print(y_adult.head())
print(X_adult.shape)

# =========================================
# 2) BANK MARKETING
# =========================================
bank = fetch_ucirepo(id=222)
X_bank = bank.data.features
y_bank = bank.data.targets
print("\n===== BANK MARKETING =====")
print(X_bank.head())
print(y_bank.head())
print(X_bank.shape)

# =========================================
# 3) HEART DISEASE
# =========================================
heart = fetch_ucirepo(id=45)
X_heart = heart.data.features
y_heart = heart.data.targets
print("\n===== HEART DISEASE =====")
print(X_heart.head())
print(y_heart.head())
print(X_heart.shape)

# =========================================
# 4) MUSHROOM
# =========================================
mushroom = fetch_ucirepo(id=73)
X_mushroom = mushroom.data.features
y_mushroom = mushroom.data.targets
print("\n===== MUSHROOM =====")
print(X_mushroom.head())
print(y_mushroom.head())
print(X_mushroom.shape)