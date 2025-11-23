"""
train_deep.py

Train deep learning model (LSTM) to predict ERCOT DART spread.
"""

import os
import sys
import pickle
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from dataloader import ERCOTDataLoader, load_features_from_aml_input
from metrics import evaluate_model, log_metrics_to_mlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LSTMRegressor(nn.Module):
    """
    LSTM-based regression model for DART spread prediction.
    
    Architecture:
    - Input: feature vector
    - LSTM layers with dropout
    - Fully connected layers
    - Output: single value (DART prediction)
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Reshape input to (batch, seq_len=1, features) for LSTM
        x = x.unsqueeze(1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out.squeeze()


def train_deep_model(
    X_train, y_train, 
    X_val, y_val,
    input_dim,
    epochs=50,
    batch_size=256,
    learning_rate=0.001
):
    """
    Train LSTM model with PyTorch.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        input_dim: Number of input features
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Returns:
        Trained PyTorch model
    """
    logger.info("="*80)
    logger.info("TRAINING LSTM MODEL")
    logger.info("="*80)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = LSTMRegressor(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.2)
    model = model.to(device)
    
    logger.info(f"Model architecture:")
    logger.info(f"  Input dim: {input_dim}")
    logger.info(f"  Hidden dim: 128")
    logger.info(f"  Num layers: 2")
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    logger.info(f"\nTraining for {epochs} epochs...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1:3d}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    logger.info(f"✓ Training complete. Best validation loss: {best_val_loss:.6f}")
    
    return model, device


def predict_with_model(model, X, device, batch_size=256):
    """Make predictions with the trained model."""
    model.eval()
    
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for batch_X, in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.append(outputs.cpu().numpy())
    
    return np.concatenate(predictions)


def main():
    """Main training pipeline."""
    try:
        logger.info("="*80)
        logger.info("DEEP LEARNING (LSTM) TRAINING PIPELINE")
        logger.info("="*80)
        logger.info(f"Start time: {datetime.now()}")
        
        # Get features path
        features_path = load_features_from_aml_input("features")
        
        # Load and prepare data
        loader = ERCOTDataLoader(features_path)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.prepare_datasets()
        
        input_dim = X_train.shape[1]
        
        # Train model
        model, device = train_deep_model(
            X_train, y_train,
            X_val, y_val,
            input_dim=input_dim,
            epochs=50,
            batch_size=256,
            learning_rate=0.001
        )
        
        # Evaluate on all sets
        logger.info("\n" + "="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)
        
        y_train_pred = predict_with_model(model, X_train, device)
        train_metrics = evaluate_model(y_train, y_train_pred, "Train")
        log_metrics_to_mlflow(train_metrics, "train")
        
        y_val_pred = predict_with_model(model, X_val, device)
        val_metrics = evaluate_model(y_val, y_val_pred, "Validation")
        log_metrics_to_mlflow(val_metrics, "val")
        
        y_test_pred = predict_with_model(model, X_test, device)
        test_metrics = evaluate_model(y_test, y_test_pred, "Test")
        log_metrics_to_mlflow(test_metrics, "test")
        
        # Save model
        logger.info("\n" + "="*80)
        logger.info("SAVING MODEL")
        logger.info("="*80)
        
        
        # Get output path from Azure ML or use local
        output_path = os.environ.get("AZUREML_OUTPUT_model", "outputs")
        os.makedirs(output_path, exist_ok=True)
        
        model_path = os.path.join(output_path, "deep_model.pt")
        
        # Save PyTorch model state dict
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'feature_columns': loader.feature_columns,
            'scaler': loader.scaler,
            'categorical_encoders': loader.categorical_encoders,
            'metrics': {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            }
        }, model_path)
        
        logger.info(f"✓ Model saved to: {model_path}")
        
        logger.info("\n" + "="*80)
        logger.info("DEEP LEARNING TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
