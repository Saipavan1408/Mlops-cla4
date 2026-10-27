"""Task 1: Train a classification pipeline on Titanic train.csv and save it as model.pkl."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Pclass", "Sex", "Embarked"]
FEATURES = CATEGORICAL + NUMERIC

df = pd.read_csv("train.csv")
print(f"Loaded train.csv: {df.shape[0]} rows, {df.shape[1]} columns")
print("Missing values:\n", df[FEATURES].isna().sum().to_string(), sep="")

X, y = df[FEATURES], df["Survived"]
# Keep PassengerId with the held-out rows so they can be used as unseen records later.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), NUMERIC),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
])
pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(n_estimators=300, max_depth=6,
                                     min_samples_leaf=2, random_state=42)),
])
pipeline.fit(X_train, y_train)

acc = accuracy_score(y_test, pipeline.predict(X_test))
print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
print(f"Test accuracy: {acc:.4f}")

joblib.dump(pipeline, "model.pkl")
# Save the unseen (held-out) records; every inference demo draws from this file.
test_out = X_test.copy()
test_out.insert(0, "PassengerId", df.loc[X_test.index, "PassengerId"])
test_out["Survived"] = y_test
test_out.to_csv("unseen_records.csv", index=False)
print("Saved model.pkl and unseen_records.csv (%d held-out records)" % len(test_out))
