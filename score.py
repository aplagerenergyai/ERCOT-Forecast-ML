"""
score.py

Real-time inference endpoint for ERCOT DART spread prediction.
Loads trained models and serves predictions via FastAPI/Uvicorn.

Supports:
- LightGBM
- XGBoost
- Deep Learning (PyTorch LSTM)

Usage:
    uvicorn score:app --host 0.0.0.0 --port 5001
"""

import os
import sys
import json
import pickle
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=os.environ.get('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

MODEL_TYPE = os.environ.get('MODEL_TYPE', 'lgbm')  # lgbm, xgb, or deep
MODEL_PATH = os.environ.get('MODEL_PATH', f'models/{MODEL_TYPE}')
LOG_DIR = os.environ.get('LOG_DIR', 'logs')

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for single or batch predictions."""
    data: List[Dict[str, Any]] = Field(
        ...,
        description="List of feature dictionaries for prediction",
        example=[{
            "TimestampHour": "2024-07-01 15:00:00",
            "SettlementPoint": "HB_HOUSTON",
            "Load_NORTH_Hourly": 50000.0,
            "Solar_Actual_Hourly": 3500.0,
            "Wind_Actual_System_Hourly": 5800.0
        }]
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[float] = Field(..., description="DART spread predictions in $/MWh")
    model_type: str = Field(..., description="Model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    count: int = Field(..., description="Number of predictions")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_type: str
    model_loaded: bool
    timestamp: str


# ============================================================================
# PyTorch Model Architecture (must match training)
# ============================================================================

class LSTMRegressor(torch.nn.Module):
    """LSTM regression model architecture."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = torch.nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = torch.nn.Linear(hidden_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        
        out = self.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out.squeeze()


# ============================================================================
# Model Loader
# ============================================================================

class ModelLoader:
    """Load and prepare trained models for inference."""
    
    def __init__(self, model_type: str, model_path: str):
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.categorical_encoders = None
        self.feature_columns = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model based on type."""
        logger.info(f"Loading {self.model_type} model from {self.model_path}")
        
        try:
            if self.model_type in ['lgbm', 'xgb']:
                # LightGBM and XGBoost use pickle
                model_file = os.path.join(self.model_path, f"{self.model_type}_model.pkl")
                
                if not os.path.exists(model_file):
                    raise FileNotFoundError(f"Model file not found: {model_file}")
                
                with open(model_file, 'rb') as f:
                    model_dict = pickle.load(f)
                
                self.model = model_dict['model']
                self.scaler = model_dict.get('scaler')
                self.categorical_encoders = model_dict.get('categorical_encoders', {})
                self.feature_columns = model_dict['feature_columns']
                
                logger.info(f"✓ {self.model_type.upper()} model loaded successfully")
                logger.info(f"  Features: {len(self.feature_columns)}")
                
            elif self.model_type == 'deep':
                # PyTorch deep learning model
                model_file = os.path.join(self.model_path, "deep_model.pt")
                
                if not os.path.exists(model_file):
                    raise FileNotFoundError(f"Model file not found: {model_file}")
                
                checkpoint = torch.load(model_file, map_location='cpu')
                
                # Recreate model architecture
                input_dim = checkpoint['input_dim']
                self.model = LSTMRegressor(
                    input_dim=input_dim,
                    hidden_dim=128,
                    num_layers=2,
                    dropout=0.2
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                self.scaler = checkpoint.get('scaler')
                self.categorical_encoders = checkpoint.get('categorical_encoders', {})
                self.feature_columns = checkpoint['feature_columns']
                
                logger.info("✓ Deep Learning model loaded successfully")
                logger.info(f"  Input dim: {input_dim}")
                logger.info(f"  Features: {len(self.feature_columns)}")
            
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
    
    def preprocess(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess input data for prediction."""
        # Ensure all feature columns are present
        missing_cols = set(self.feature_columns) - set(data.columns)
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}. Filling with 0.")
            for col in missing_cols:
                data[col] = 0.0
        
        # Select and order features
        X = data[self.feature_columns].copy()
        
        # Handle categorical encoding (if needed)
        for col, encoder in self.categorical_encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[[col]])[col]
        
        # Fill missing values
        X = X.fillna(0)
        
        # Standardize
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X.values)
        else:
            X_scaled = X.values
        
        return X_scaled
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        try:
            if self.model_type in ['lgbm', 'xgb']:
                predictions = self.model.predict(X)
            
            elif self.model_type == 'deep':
                X_tensor = torch.FloatTensor(X)
                with torch.no_grad():
                    predictions = self.model(X_tensor).numpy()
            
            return predictions
        
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="ERCOT DART Spread Prediction API",
    description="Real-time predictions for Day-Ahead vs Real-Time price spread",
    version="1.0.0"
)

# Global model loader
model_loader: Optional[ModelLoader] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model_loader
    
    logger.info("="*80)
    logger.info("STARTING ERCOT DART PREDICTION SERVICE")
    logger.info("="*80)
    logger.info(f"Model Type: {MODEL_TYPE}")
    logger.info(f"Model Path: {MODEL_PATH}")
    logger.info(f"Log Level: {os.environ.get('LOG_LEVEL', 'INFO')}")
    
    try:
        model_loader = ModelLoader(MODEL_TYPE, MODEL_PATH)
        logger.info("✓ Service ready for predictions")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {str(e)}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "service": "ERCOT DART Prediction API",
        "status": "running",
        "model_type": MODEL_TYPE,
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_loader is not None else "unhealthy",
        model_type=MODEL_TYPE,
        model_loaded=model_loader is not None,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/score", response_model=PredictionResponse)
async def score(request: PredictionRequest):
    """
    Score endpoint for real-time predictions.
    
    Accepts a list of feature dictionaries and returns DART predictions.
    """
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Received prediction request with {len(request.data)} records")
        
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        # Preprocess
        X = model_loader.preprocess(df)
        
        # Predict
        predictions = model_loader.predict(X)
        
        # Convert to list
        pred_list = predictions.tolist()
        
        logger.info(f"Generated {len(pred_list)} predictions")
        
        return PredictionResponse(
            predictions=pred_list,
            model_type=MODEL_TYPE,
            timestamp=datetime.utcnow().isoformat(),
            count=len(pred_list)
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model/info", response_model=Dict[str, Any])
async def model_info():
    """Get model information."""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": MODEL_TYPE,
        "feature_count": len(model_loader.feature_columns),
        "feature_columns": model_loader.feature_columns,
        "has_scaler": model_loader.scaler is not None,
        "categorical_encoders": list(model_loader.categorical_encoders.keys())
    }


# ============================================================================
# Main Entry Point (for testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get('PORT', 5001))
    
    uvicorn.run(
        "score:app",
        host="0.0.0.0",
        port=port,
        log_level=os.environ.get('LOG_LEVEL', 'info').lower()
    )

