from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "API is working"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Extract values in correct order
        store = data["Store"]
        holiday_flag = data["Holiday_Flag"]
        temperature = data["Temperature"]
        fuel_price = data["Fuel_Price"]
        cpi = data["CPI"]
        unemployment = data["Unemployment"]
        day = data["day"]
        month = data["month"]
        year = data["year"]

        # Create input array in SAME ORDER as training
        input_data = np.array([[store, holiday_flag, temperature, fuel_price, cpi, unemployment, day, month, year]])

        prediction = model.predict(input_data)

        return {"prediction": float(prediction[0])}

    except Exception as e:
        return {"error": str(e)}