import numpy as np
from sklearn.neural_network import MLPRegressor
from typing import List, Dict, Optional, Tuple, Union
import joblib

class QuantumErrorNeuralNet:
    """
    Neural network for quantum error mitigation.
    
    This model takes the output distribution of a noisy quantum circuit
    (bitstring histogram) and predicts what the true noise-free distribution 
    would have been.
    
    Input nodes: Each corresponds to a possible bitstring outcome from the circuit
                 with normalized frequency values
    Output nodes: Predicted "true" distribution that would occur without noise
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_layer_sizes: Tuple[int, ...] = (100,),
                 activation: str = 'relu',
                 solver: str = 'adam',
                 alpha: float = 0.0001,
                 batch_size: Union[str, int] = 'auto',
                 learning_rate: str = 'constant',
                 learning_rate_init: float = 0.001,
                 max_iter: int = 200,
                 random_state: Optional[int] = None):
        """
        Initialize the quantum error neural network.
        
        Args:
            input_size: Number of input features (possible bitstrings from circuit)
            hidden_layer_sizes: Number of neurons in each hidden layer
            activation: Activation function for hidden layers
            solver: Solver algorithm for weight optimization
            alpha: L2 penalty parameter
            batch_size: Size of minibatches for stochastic optimizers
            learning_rate: Learning rate schedule for weight updates
            learning_rate_init: Initial learning rate for weight updates
            max_iter: Maximum number of iterations
            random_state: Seed for reproducibility
        """
        self.input_size = input_size
        self.output_size = input_size  # Same size as input for distribution prediction
        
        # Initialize the neural network
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state
        )
        
    def train(self, noisy_distributions: np.ndarray, true_distributions: np.ndarray) -> float:
        """
        Train the neural network model.
        
        Args:
            noisy_distributions: Array of shape (n_samples, input_size) containing 
                                noisy output distributions.
            true_distributions: Array of shape (n_samples, input_size) containing 
                               the corresponding noise-free distributions.
                               
        Returns:
            Score of the model on the training data.
        """
        self.model.fit(noisy_distributions, true_distributions)
        return self.model.score(noisy_distributions, true_distributions)
    
    def predict(self, noisy_distribution: np.ndarray) -> np.ndarray:
        """
        Predict the true distribution from a noisy one.
        
        Args:
            noisy_distribution: Array of shape (n_samples, input_size) containing
                               the noisy output distribution(s).
                               
        Returns:
            Predicted true distribution(s).
        """
        predictions = self.model.predict(noisy_distribution)
        
        # Ensure all probabilities are non-negative
        predictions = np.maximum(predictions, 0)
        
        # Normalize to ensure sum is 1 for each sample
        row_sums = predictions.sum(axis=1, keepdims=True)
        normalized_predictions = predictions / row_sums
        
        return normalized_predictions
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model to.
        """
        joblib.dump(self, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'QuantumErrorNeuralNet':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from.
            
        Returns:
            Loaded QuantumErrorNeuralNet object.
        """
        return joblib.load(filepath)
    
    @classmethod
    def from_num_qubits(cls, 
                       num_qubits: int, 
                       hidden_layer_sizes: Tuple[int, ...] = None,
                       **kwargs) -> 'QuantumErrorNeuralNet':
        """
        Create a model based on the number of qubits.
        
        Args:
            num_qubits: Number of qubits in the circuit.
            hidden_layer_sizes: The number of neurons in each hidden layer.
            **kwargs: Additional arguments to pass to the constructor.
            
        Returns:
            Initialized QuantumErrorNeuralNet.
        """
        input_size = 2**num_qubits
        
        # Default hidden layer sizes if not specified
        if hidden_layer_sizes is None:
            # Scale hidden layer size with the input size
            hidden_layer_sizes = (max(100, input_size * 2), max(50, input_size))
            
        return cls(input_size, hidden_layer_sizes, **kwargs)
    
    @staticmethod
    def format_counts_to_distribution(counts: Dict[str, int], num_qubits: int) -> np.ndarray:
        """
        Convert a counts dictionary from a quantum circuit to a normalized distribution array.
        
        Args:
            counts: Dictionary mapping bitstrings to counts.
            num_qubits: Number of qubits in the circuit.
            
        Returns:
            Normalized distribution array of shape (1, 2^num_qubits).
        """
        distribution = np.zeros(2**num_qubits)
        total_counts = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Convert bitstring to index (binary to decimal)
            index = int(bitstring, 2)
            distribution[index] = count / total_counts
            
        return distribution.reshape(1, -1)  # Reshape for model input
    
    @staticmethod
    def distribution_to_counts(distribution: np.ndarray, total_shots: int = 1000) -> Dict[str, int]:
        """
        Convert a predicted distribution back to a counts dictionary.
        
        Args:
            distribution: Distribution array of shape (2^num_qubits,) or (1, 2^num_qubits).
            total_shots: Total number of shots to simulate.
            
        Returns:
            Dictionary mapping bitstrings to counts.
        """
        # Ensure distribution is a 1D array
        if len(distribution.shape) > 1:
            distribution = distribution.flatten()
            
        num_qubits = int(np.log2(len(distribution)))
        counts = {}
        
        # Round counts to integers based on distribution probabilities
        raw_counts = (distribution * total_shots).astype(int)
        
        # Adjust to ensure the total count is exactly total_shots
        adjustment = total_shots - np.sum(raw_counts)
        if adjustment > 0:
            # Add 1 to the highest probability indices that were rounded down
            indices = np.argsort(distribution - raw_counts/total_shots)[-adjustment:]
            for idx in indices:
                raw_counts[idx] += 1
        
        for i in range(len(distribution)):
            count = raw_counts[i]
            if count > 0:
                # Convert index to bitstring
                bitstring = format(i, f'0{num_qubits}b')
                counts[bitstring] = count
                
        return counts