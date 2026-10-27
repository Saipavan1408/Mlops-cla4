"""Task 2: Online inference - one unseen passenger, one prediction."""
import joblib
import pandas as pd

FEATURES = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"]
model = joblib.load("model.pkl")
unseen = pd.read_csv("unseen_records.csv")

record = unseen.iloc[[0]]
pred = int(model.predict(record[FEATURES])[0])
prob = float(model.predict_proba(record[FEATURES])[0, 1])

print("=== ONLINE INFERENCE (single record) ===")
print(record[["PassengerId"] + FEATURES].to_string(index=False))
print(f"\nPrediction : {pred} ({'Survived' if pred else 'Did not survive'})")
print(f"P(survive) : {prob:.3f}")
print(f"Actual     : {int(record['Survived'].iloc[0])}")
