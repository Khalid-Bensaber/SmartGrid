from fastapi import FastAPI
from smartgrid.service import predict_conso

app = FastAPI()

@app.get("/")
def root():
    return {"message": "SmartGrid API running"}

@app.post("/predict")
def predict():
    return predict_conso()
