"""Python client to test the FastAPI service (start the server first)."""
import requests

BASE = "http://127.0.0.1:8000"

print("GET /health ->", requests.get(f"{BASE}/health").json())

valid = {"Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1,
         "Parch": 0, "Fare": 71.28, "Embarked": "C"}
r = requests.post(f"{BASE}/predict", json=valid)
print("\nPOST /predict (valid)   ->", r.status_code, r.json())

invalid = {"Pclass": 5, "Sex": "unknown", "Age": -3, "SibSp": 1, "Parch": 0, "Fare": "abc"}
r = requests.post(f"{BASE}/predict", json=invalid)
print("\nPOST /predict (invalid) ->", r.status_code)
for err in r.json()["detail"]:
    print("   -", ".".join(map(str, err["loc"][1:])), ":", err["msg"])
