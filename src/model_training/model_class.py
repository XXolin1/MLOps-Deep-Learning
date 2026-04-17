import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Callable, Optional, Union
from tqdm.auto import tqdm
import copy
from sklearn.metrics import roc_auc_score

class PyTorchMLP(nn.Module):
    """
    Multi-Layer Perceptron using PyTorch with Keras-like API
    
    Features:
    - Parametric hidden layers
    - Customizable activation functions
    - Keras-style fit() and predict() methods
    - Automatic numpy/pandas to tensor conversion
    - Support for various loss functions and optimizers
    - Dropout, L2 Regularization, and Early Stopping
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_layers: List[int],
                 output_size: int,
                 activation: Union[str, Callable] = 'relu',
                 output_activation: Union[str, Callable] = 'sigmoid',
                 loss_function: Union[str, Callable] = 'mse',
                 optimizer: str = 'adam',
                 learning_rate: float = 0.01,
                 epochs: int = 100,
                 batch_size: int = 32,
                 metrics: str = 'loss',
                 dropout: float = 0.0,
                 l2_reg: float = 0.0,
                 early_stopping: bool = False,
                 early_stopping_patience: int = None,
                 early_stopping_min_delta: float = None,
                 device: str = 'cpu'):
        """
        Initialize the MLP.
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes, e.g., [64, 32, 16]
            output_size: Number of output units
            activation: Activation function for hidden layers ('relu', 'tanh', 'sigmoid')
            output_activation: Activation for output layer ('sigmoid', 'softmax', 'linear')
            loss_function: Loss function ('mse', 'bce', 'cross_entropy')
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs (default: 100)
            batch_size: Batch size for training (default: 32)
            metric: Metric name to compute during training
            dropout: Dropout probability (0 to 1) applied after each hidden layer
            l2_reg: L2 regularization / weight decay term
            early_stopping: Whether to use early stopping based on validation loss
            early_stopping_patience: Number of epochs to wait for improvement before early stopping
            early_stopping_min_delta: Minimum change in validation loss to qualify as an improvement
            device: 'cpu' or 'cuda'
        """
        super(PyTorchMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.metrics = metrics.lower()
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = torch.device(device)
        
        # Store activation functions
        self.activation_fn = self._get_activation(activation)
        self.output_activation_fn = self._get_activation(output_activation)
        self.output_activation_name = output_activation.__name__ if callable(output_activation) else output_activation
        
        # Build network
        layers = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # Don't add activation after output layer
                match activation:
                    case 'relu':
                        layers.append(nn.ReLU())
                    case 'leakyrelu':
                        layers.append(nn.LeakyReLU())
                    case 'tanh':
                        layers.append(nn.Tanh())
                    case 'sigmoid':
                        layers.append(nn.Sigmoid())
                    case _:
                        pass  # Default to no activation if unrecognized
                
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
        
        self.network = nn.Sequential(*layers)
        self.to(self.device)
        
        # Loss function
        self.loss_fn = self._get_loss_function(loss_function)
        
        # Optimizer
        self.optimizer = self._get_optimizer(optimizer)
        
        # Training history
        self.history = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'auc': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1_score': [], 'val_auc': []
        }
        
    def _compute_metrics(self, y_pred: torch.Tensor, y_batch: torch.Tensor, metric_name: str) -> float:
        """Helper to compute evaluation metrics."""
        if metric_name == 'loss' or self.output_activation_name.lower() == 'linear':
            return self.loss_fn(y_pred, y_batch).item()
            
        with torch.no_grad():
            y_pred_np = y_pred.cpu().numpy()
            y_batch_np = y_batch.cpu().numpy()
            
            match self.output_activation_name.lower():
                case 'sigmoid':
                    preds = (y_pred >= 0.5).float()
                case 'tanh':
                    preds = (y_pred >= 0.5).float()
                case 'softmax':
                    preds = torch.argmax(y_pred, dim=1, keepdim=True).float()
                    if y_batch.dim() > 1 and y_batch.shape[1] > 1:
                        y_batch = torch.argmax(y_batch, dim=1, keepdim=True).float()
                case _:
                    return self.loss_fn(y_pred, y_batch).item()
            
            match metric_name:
                case 'accuracy':
                    return (preds == y_batch).float().mean().item()
                case 'auc':
                    # ROC AUC requires both classes (0 and 1) to be present in the batch
                    if len(np.unique(y_batch_np)) > 1:
                        return roc_auc_score(y_batch_np, y_pred_np)
                    return 0.5 # Default for single class batches
            
            # True Positives, False Positives, False Negatives wrapper
            tp = ((preds == 1) & (y_batch == 1)).float().sum()
            fp = ((preds == 1) & (y_batch == 0)).float().sum()
            fn = ((preds == 0) & (y_batch == 1)).float().sum()
            
            match metric_name:
                case 'precision':
                    return (tp / (tp + fp + 1e-8)).item()
                case 'recall':
                    return (tp / (tp + fn + 1e-8)).item()
                case 'f1_score':
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    return (2 * precision * recall / (precision + recall + 1e-8)).item()
                
        return self.loss_fn(y_pred, y_batch).item()

    def _get_activation(self, activation: Union[str, Callable]) -> Callable:
        """Get activation function by name or return callable."""
        if callable(activation):
            return activation
        
        activations = {
            'relu': torch.relu,
            'leakyrelu': torch.nn.functional.leaky_relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'softmax': lambda x: torch.softmax(x, dim=1),
            'linear': lambda x: x,
        }
        return activations.get(activation.lower(), torch.relu)
    
    def _get_loss_function(self, loss: Union[str, Callable]) -> Callable:
        """Get loss function by name or return callable."""
        if callable(loss):
            return loss
        
        loss_map = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'bce': nn.BCELoss(),
            'bce_logits': nn.BCEWithLogitsLoss(),
            'cross_entropy': nn.CrossEntropyLoss(),
        }
        return loss_map.get(loss.lower(), nn.MSELoss())
    
    def _get_optimizer(self, optimizer_name: str) -> optim.Optimizer:
        """Get optimizer."""
        optimizer_map = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
            'adamw': optim.AdamW,
        }
        OptimizerClass = optimizer_map.get(optimizer_name.lower(), optim.Adam)
        return OptimizerClass(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)

    def _get_model_device(self) -> torch.device:
        """Return the current device used by the model parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self.device
    
    def _to_tensor(self, data: Union[np.ndarray, 'pd.DataFrame', torch.Tensor]) -> torch.Tensor:
        """Convert numpy array, pandas DataFrame, or list to tensor."""
        device = self._get_model_device()

        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif hasattr(data, 'to_numpy'):  # pandas DataFrame
            return torch.FloatTensor(data.to_numpy().copy()).to(device)
        else:  # numpy array or list
            return torch.FloatTensor(np.asarray(data)).to(device)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            X: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor with applied output activation
        """
        output = self.network(X)
        
        # Apply output activation
        if self.output_activation_name.lower() != 'linear':
            output = self.output_activation_fn(output)

        return output
    
    def backward(self, loss: torch.Tensor) -> None:
        """
        Backward pass (gradient computation and parameter update).
        
        Args:
            loss: Scalar loss value to backpropagate
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def fit(self,
            X: Union[np.ndarray, 'pd.DataFrame'],
            y: Union[np.ndarray, 'pd.Series'],
            X_val: Optional[Union[np.ndarray, 'pd.DataFrame']] = None,
            y_val: Optional[Union[np.ndarray, 'pd.Series']] = None) -> dict:
        """
        Train the model on the data (Keras-style fit method).
        
        Args:
            X: Training input features (numpy, pandas, or tensor)
            y: Training target labels (numpy, pandas, or tensor)
            X_val: Optional validation input features for early stopping
            y_val: Optional validation target labels for early stopping
            
        Returns:
            Dictionary with training history
        """
        # Convert to tensors
        X_tensor = self._to_tensor(X)
        y_tensor = self._to_tensor(y)
        
        # Reshape y if needed
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(1)
        
        # Create data loader
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Reset training history for fresh fit
        self.history = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'auc': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1_score': [], 'val_auc': []
        }
        
        metrics_list = ['loss', 'accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        do_validation = X_val is not None and y_val is not None
        is_early_stopping_configured = self.early_stopping and self.early_stopping_min_delta is not None and self.early_stopping_patience is not None
        
        if do_validation:
            # Pre-compute validation loader ONCE to avoid massive overhead
            X_val_tensor = self._to_tensor(X_val)
            y_val_tensor = self._to_tensor(y_val)
            if y_val_tensor.dim() == 1:
                y_val_tensor = y_val_tensor.unsqueeze(1)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            best_val_metric = float('inf') if self.metrics == 'loss' else float('-inf')
            best_weights = None
            epochs_no_improve = 0
        
        if self.early_stopping and not do_validation:
            print("Warning: Early stopping enabled but no validation data provided.")
        
        if self.early_stopping and do_validation and not is_early_stopping_configured:
            print("Warning: Early stopping enabled but patience or min_delta not configured.")

        if tqdm is not None:
            pbar = tqdm(total=self.epochs, desc='Training')
        else:
            pbar = None

        for epoch in range(self.epochs):
            self.train()
            epoch_metrics = {m: 0.0 for m in metrics_list}
            num_batches = 0
            
            for X_batch, y_batch in train_loader:
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                loss = self.loss_fn(y_pred, y_batch)
                
                # Backward pass
                self.backward(loss)
                
                # Collect metrics
                epoch_metrics['loss'] += loss.item()
                for m in metrics_list[1:]:
                    epoch_metrics[m] += self._compute_metrics(y_pred, y_batch, m)
                
                num_batches += 1
            
            # Average training metrics
            for m in metrics_list:
                self.history[m].append(epoch_metrics[m] / num_batches)
            
            # Validation
            if do_validation:
                self.eval()
                val_epoch_metrics = {f'val_{m}': 0.0 for m in metrics_list}
                val_batches = 0
                
                with torch.no_grad():
                    for X_v, y_v in val_loader:
                        v_pred = self.forward(X_v)
                        v_loss = self.loss_fn(v_pred, y_v)
                        
                        val_epoch_metrics['val_loss'] += v_loss.item()
                        for m in metrics_list[1:]:
                            val_epoch_metrics[f'val_{m}'] += self._compute_metrics(v_pred, y_v, m)
                        val_batches += 1
                
                # Average validation metrics
                for m in metrics_list:
                    self.history[f'val_{m}'].append(val_epoch_metrics[f'val_{m}'] / val_batches)
                
                # Early stopping logic
                if is_early_stopping_configured:
                    current_val_metric = self.history[f'val_{self.metrics}'][-1]
                    
                    if self.metrics == 'loss':
                        is_better = current_val_metric < best_val_metric - self.early_stopping_min_delta
                    else:
                        is_better = current_val_metric > best_val_metric + self.early_stopping_min_delta
                    
                    if is_better:
                        best_val_metric = current_val_metric
                        best_weights = copy.deepcopy(self.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        
                    if epochs_no_improve >= self.early_stopping_patience:
                        if pbar is not None:
                            pbar.set_description('Stopped Early')
                            pbar.update((epoch + 1) - pbar.n)
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        self.load_state_dict(best_weights)
                        break
            
            if pbar is not None:
                postfix = {'loss': f"{self.history['loss'][-1]:.4f}"}
                if do_validation:
                    postfix['val_loss'] = f"{self.history['val_loss'][-1]:.4f}"
                    if self.metrics != 'loss':
                        postfix[f'val_{self.metrics}'] = f"{self.history['val_' + self.metrics][-1]:.4f}"
                pbar.set_postfix(postfix)
                pbar.update(1)
            
        if pbar is not None:
            pbar.close()
            
        # Restore best weights if we didn't early stop but generated weights
        if is_early_stopping_configured and best_weights is not None and epochs_no_improve < self.early_stopping_patience:
            self.load_state_dict(best_weights)

        return self.history
    
    def predict(self, X: Union[np.ndarray, 'pd.DataFrame']) -> np.ndarray:
        """
        Make predictions on new data (Keras-style predict method).
        
        Args:
            X: Input features (numpy, pandas, or tensor)
            
        Returns:
            Predictions as numpy array
        """
        # Convert to tensor
        X_tensor = self._to_tensor(X)
        
        # Create data loader
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Make predictions
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for X_batch, in loader:
                y_pred = self.forward(X_batch)
                predictions.append(y_pred.cpu().numpy())
        
        # Concatenate all predictions
        predictions = np.vstack(predictions)
        
        return predictions
    
    def show_evolution(self) -> None:
        """Display training evolution graph (loss and metrics) in a 3x2 grid."""
        metrics = [m for m in self.history.keys() if not m.startswith('val_')]
        
        # Fixed 3x2 grid
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()
            
        for i, metric in enumerate(metrics):
            if i < len(axes):
                axes[i].plot(self.history[metric], label='Train')
                if f'val_{metric}' in self.history:
                    axes[i].plot(self.history[f'val_{metric}'], label='Val')
                axes[i].set_title(f'Model {metric.capitalize()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].legend()
                axes[i].grid(True)
            
        plt.tight_layout()
        plt.show()

    def evaluate(self,
                 X: Union[np.ndarray, 'pd.DataFrame'],
                 y: Union[np.ndarray, 'pd.Series']) -> float:
        """
        Evaluate model on test data.
        
        Args:
            X: Test input features
            y: Test target labels
            
        Returns:
            Average loss on test data
        """
        X_tensor = self._to_tensor(X)
        y_tensor = self._to_tensor(y)
        
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                y_pred = self.forward(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches