# ML Model Deployment Strategies and Model Serving using FastAPI

Trains a Titanic survival classifier once, saves the full pipeline (`model.pkl`), and reuses it for
online, batch and streaming inference and for a FastAPI REST service.

**Dataset:** Titanic - Machine Learning from Disaster (Kaggle), `train.csv`
https://www.kaggle.com/competitions/titanic/data

## Files
| File | Purpose |
|---|---|
| `task1_train.py` | Preprocess, train, evaluate (accuracy), save `model.pkl` and `unseen_records.csv` |
| `task2_online.py` | Online inference: 1 unseen record |
| `task3_batch.py` | Batch inference: 20 unseen records -> `batch_predictions.csv` |
| `task4_streaming.py` | Streaming inference: 10 unseen records, 1 per second |
| `app.py` | FastAPI app (`GET /health`, `POST /predict`, Pydantic validation) |
| `client.py` | Python client testing the API (valid + invalid request) |
| `report.pdf` | Task 6 observations |
| `screenshots/` | Output screenshots |

## Run
```bash
pip install -r requirements.txt
# place Kaggle train.csv in this folder
python task1_train.py
python task2_online.py
python task3_batch.py
python task4_streaming.py
uvicorn app:app --reload          # Swagger UI: http://127.0.0.1:8000/docs
python client.py                  # in a second terminal
```

## Results
- Model: Random Forest in a scikit-learn `Pipeline` (imputation, scaling, one-hot encoding).
- Split: 80/20 stratified (712 train / 179 test). **Test accuracy: 0.8101**
- Unseen records come from the 179 held-out rows (never used for training).
- Online: PassengerId 566 -> Did not survive (p=0.130), matches actual.
- Batch: 20 records, 17/20 match actual labels, saved to `batch_predictions.csv`.
- Streaming: 10 records, one prediction per second.
- API: `/health` returns `{"status":"ok","model_loaded":true}`; valid `/predict` returns 200;
  invalid input returns 422 with Pydantic error details.
