
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Function to load the model


def load_model(model_path: str):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            logger.info("Model loaded successfully from %s.", model_path)
        return model
    except FileNotFoundError:
        logger.error("Model file not found at %s.", model_path)
        raise HTTPException(status_code=500, detail="Model file not found.")
    except Exception as e:
        logger.exception("An error occurred while loading the model.")
        raise HTTPException(status_code=500, detail="Model loading failed.")


# Initialize FastAPI app and load the model
app = FastAPI()

#CORS for Local Development and Connection


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

model_path = os.getenv("MODEL_PATH", "FloodRisk.pkl")
model = load_model(model_path)

# Define the input data model using Pydantic


class FloodData(BaseModel):
    MonsoonIntensity: float
    TopographyDrainage: float
    RiverManagement: float
    Deforestation: float
    Urbanization: float
    ClimateChange: float
    DamsQuality: float
    Siltation: float
    AgriculturalPractices: float
    Encroachments: float
    IneffectiveDisasterPreparedness: float
    DrainageSystems: float
    CoastalVulnerability: float
    Landslides: float
    Watersheds: float
    DeterioratingInfrastructure: float
    PopulationScore: float
    WetlandLoss: float
    InadequatePlanning: float
    PoliticalFactors: float


# List of feature names to maintain input consistency
FEATURE_NAMES = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation", "AgriculturalPractices",
    "Encroachments", "IneffectiveDisasterPreparedness", "DrainageSystems", "CoastalVulnerability",
    "Landslides", "Watersheds", "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
    "InadequatePlanning", "PoliticalFactors"
]


@app.get("/")
async def index():
    return {'message': 'Hello Users'}


@app.post("/predict")
async def predict_flood(data: FloodData):
    # Convert input data to a pandas DataFrame with feature names
    input_data_df = pd.DataFrame([data.dict()], columns=FEATURE_NAMES)

    # Make prediction using the loaded model
    try:
        prediction = model.predict(input_data_df)
        prediction_proba = model.predict_proba(input_data_df)
        logger.info("Prediction made successfully.")
    except Exception as e:
        logger.exception("Model prediction failed.")
        raise HTTPException(status_code=500, detail="Model prediction failed.")

    # Return the prediction result
    return {
        "prediction": int(prediction[0]),
        "prediction_probability": prediction_proba[0].tolist()
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
