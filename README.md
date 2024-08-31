
# Live Webservice
## https://floodapi.onrender.com/docs 


# Flood Risk Prediction API

This API is built using FastAPI to predict flood risks based on various environmental and infrastructural factors. The model used for prediction is loaded from a pickle file and uses features like Monsoon Intensity, Urbanization, Climate Change, and more.

## Features

- Load a pre-trained model from a specified path.
- Predict flood risk based on input data with 20 different features.
- Structured logging for better tracking and debugging.
- CORS support for local development and external connections.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- FastAPI
- Uvicorn
- Pydantic
- Pandas
- Pickle
- Python-dotenv

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/flood-risk-prediction-api.git
    cd flood-risk-prediction-api
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the root directory and add the following environment variables:

    ```env
    MODEL_PATH=path/to/your/FloodRisk.pkl
    HOST=127.0.0.1
    PORT=8000
    ```

4. Place your trained model (e.g., `FloodRisk.pkl`) in the specified path or update the `MODEL_PATH` in the `.env` file.

### Running the API

Run the API using Uvicorn:

```bash
uvicorn main:app --reload

