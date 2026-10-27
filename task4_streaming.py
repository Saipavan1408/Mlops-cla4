"""Task 4: Streaming inference - 10 unseen records, one at a time, 1 second apart."""
import time
import joblib
import pandas as pd

FEATURES = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"]
model = joblib.load("model.pkl")
unseen = pd.read_csv("unseen_records.csv")
stream = unseen.iloc[21:31]               # 10 records (distinct from online + batch)

print("=== STREAMING INFERENCE (1 record / second) ===")
for i, (_, row) in enumerate(stream.iterrows(), start=1):
    rec = row[FEATURES].to_frame().T
    pred = int(model.predict(rec)[0])
    prob = float(model.predict_proba(rec)[0, 1])
    print(f"[{time.strftime('%H:%M:%S')}] #{i:02d} PassengerId={int(row['PassengerId']):<4} "
          f"{row['Sex']:<6} Pclass={int(row['Pclass'])} Age={row['Age']} -> "
          f"{'Survived' if pred else 'Did not survive':<15} (p={prob:.3f})")
    time.sleep(1)
print("Stream finished.")
