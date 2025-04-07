from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import joblib
import os
from io import StringIO
import uvicorn

from src.config import MODEL_PATH

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Enrollment Prediction API",
    description="API for predicting employee insurance enrollment probability",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model


# Helper function to load the model
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        model_info = joblib.load(MODEL_PATH)
        return model_info
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


# Define input data models
class Employee(BaseModel):
    employee_id: int = Field(..., example=20001)
    age: int = Field(..., example=35, ge=18, le=100)
    gender: str = Field(..., example="Male")
    marital_status: str = Field(..., example="Married")
    salary: float = Field(..., example=75000.0, ge=0)
    employment_type: str = Field(..., example="Full-time")
    region: str = Field(..., example="West")
    has_dependents: str = Field(..., example="Yes")
    tenure_years: float = Field(..., example=5.5, ge=0)


class EmployeeBatch(BaseModel):
    employees: List[Employee]


class PredictionResponse(BaseModel):
    employee_id: int
    enrollment_probability: float
    predicted_enrollment: bool


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


# Routes
@app.get("/")
async def root():
    return {"message": "Insurance Enrollment Prediction API. Use /docs for API documentation."}


@app.get("/health")
async def health_check():
    # Check if model can be loaded
    try:
        model_info = load_model()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(employee: Employee):
    model_info = load_model()

    # Create DataFrame from input
    data = pd.DataFrame([employee.dict()])

    # Make prediction
    try:
        # Preprocess the data
        processed_data = model_info['preprocessor'].transform(data)

        # Get prediction probability
        probability = model_info['model'].predict_proba(processed_data)[0, 1]

        # Determine prediction (1 if probability >= 0.5, else 0)
        prediction = probability >= 0.5

        return PredictionResponse(
            employee_id=employee.employee_id,
            enrollment_probability=float(probability),
            predicted_enrollment=bool(prediction)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(employees: EmployeeBatch):
    model_info = load_model()

    # Create DataFrame from input
    data = pd.DataFrame([emp.dict() for emp in employees.employees])

    # Make predictions
    try:
        # Preprocess the data
        processed_data = model_info['preprocessor'].transform(data)

        # Get prediction probabilities
        probabilities = model_info['model'].predict_proba(processed_data)[:, 1]

        # Determine predictions
        predictions = probabilities >= 0.5

        # Create response
        response = [
            PredictionResponse(
                employee_id=int(employee_id),
                enrollment_probability=float(prob),
                predicted_enrollment=bool(pred)
            )
            for employee_id, prob, pred in zip(data['employee_id'], probabilities, predictions)
        ]

        return BatchPredictionResponse(predictions=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/predict-from-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    model_info = load_model()

    # Check if file is CSV
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # Read CSV content
    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode())
        data = pd.read_csv(csv_data)

        # Check if required columns are present
        required_columns = [
            'employee_id', 'age', 'gender', 'marital_status', 'salary',
            'employment_type', 'region', 'has_dependents', 'tenure_years'
        ]

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )

        # Preprocess the data
        processed_data = model_info['preprocessor'].transform(data)

        # Get prediction probabilities
        probabilities = model_info['model'].predict_proba(processed_data)[:, 1]

        # Determine predictions
        predictions = probabilities >= 0.5

        # Add predictions to the data
        data['enrollment_probability'] = probabilities
        data['predicted_enrollment'] = predictions

        # Convert to list of dictionaries for the response
        result = data.to_dict(orient='records')

        return {"predictions": result, "total_records": len(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")


@app.get("/model-info")
async def get_model_info():
    model_info = load_model()

    # Extract useful information about the model
    info = {
        "model_type": "Logistic Regression",
        "feature_count": len(model_info.get('feature_names', [])),
        "feature_names": model_info.get('feature_names', []),
        "accuracy": model_info.get('accuracy', None),
        "roc_auc": model_info.get('roc_auc', None),
    }

    return info


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)