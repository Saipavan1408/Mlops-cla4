"""Task 3: Batch inference - 20 unseen passengers predicted together."""
import joblib
import pandas as pd

FEATURES = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"]
model = joblib.load("model.pkl")
unseen = pd.read_csv("unseen_records.csv")

batch = unseen.iloc[1:21].copy()          # 20 records (distinct from the online one)
batch["Predicted"] = model.predict(batch[FEATURES])           # single call for all 20
batch["P_survive"] = model.predict_proba(batch[FEATURES])[:, 1].round(3)
batch.to_csv("batch_predictions.csv", index=False)

print("=== BATCH INFERENCE (20 records) ===")
print(batch[["PassengerId", "Pclass", "Sex", "Age", "Predicted", "P_survive", "Survived"]]
      .to_string(index=False))
print(f"\nSaved {len(batch)} predictions to batch_predictions.csv")
print(f"Matches with actual label: {(batch['Predicted'] == batch['Survived']).sum()}/20")
