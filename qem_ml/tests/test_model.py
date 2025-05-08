import pytest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import the module
sys.path.append(str(Path(__file__).parent.parent))
from qem_ml.models.qe_neural_net import QuantumErrorNeuralNet


class TestQuantumErrorNeuralNet:
    
    def test_initialization(self):
        """Test basic initialization of the neural network."""
        # Basic initialization
        model = QuantumErrorNeuralNet(input_size=8)
        assert model.input_size == 8
        assert model.output_size == 8
        assert model.model is not None
        
        # Initialization with custom parameters
        model = QuantumErrorNeuralNet(
            input_size=16,
            hidden_layer_sizes=(32, 16),
            activation='tanh',
            max_iter=500,
            random_state=42
        )
        assert model.input_size == 16
        assert model.model.hidden_layer_sizes == (32, 16)
        assert model.model.activation == 'tanh'
        assert model.model.max_iter == 500
        assert model.model.random_state == 42
        
    def test_from_num_qubits(self):
        """Test initialization from number of qubits."""
        # 2 qubits should have 4 input/output nodes
        model = QuantumErrorNeuralNet.from_num_qubits(num_qubits=2)
        assert model.input_size == 4
        assert model.output_size == 4
        
        # 3 qubits should have 8 input/output nodes
        model = QuantumErrorNeuralNet.from_num_qubits(
            num_qubits=3,
            hidden_layer_sizes=(20, 10),
            activation='tanh'
        )
        assert model.input_size == 8
        assert model.output_size == 8
        assert model.model.hidden_layer_sizes == (20, 10)
        assert model.model.activation == 'tanh'
        
    def test_train_and_predict(self):
        """Test training and prediction functionality."""
        # Create simple test data for a 2-qubit system
        np.random.seed(42)
        
        # Generate synthetic data
        true_distributions = np.array([
            [0.25, 0.25, 0.25, 0.25],  # Uniform distribution
            [0.4, 0.3, 0.2, 0.1],      # Biased distribution
            [0.1, 0.1, 0.7, 0.1]       # Another distribution
        ])
        
        # Create noisy versions by adding noise and renormalizing
        noise = np.random.normal(0, 0.05, true_distributions.shape)
        noisy_distributions = true_distributions + noise
        # Ensure non-negative values
        noisy_distributions = np.maximum(noisy_distributions, 0)
        # Normalize
        row_sums = noisy_distributions.sum(axis=1, keepdims=True)
        noisy_distributions = noisy_distributions / row_sums
        
        # Create and train the model - Increase max_iter to help with convergence
        model = QuantumErrorNeuralNet(
            input_size=4, 
            hidden_layer_sizes=(20, 10), 
            random_state=42,
            max_iter=1000,  # Increase iterations for better convergence
            learning_rate_init=0.01  # Adjust learning rate
        )
        score = model.train(noisy_distributions, true_distributions)
        
        # Check that training produced a valid score
        # Note: R² scores can be negative if model performs worse than a horizontal line
        assert isinstance(score, float)
        assert not np.isnan(score)
        
        # Test prediction
        predictions = model.predict(noisy_distributions)
        
        # Check shape and normalization
        assert predictions.shape == true_distributions.shape
        assert np.allclose(predictions.sum(axis=1), 1.0)  # Sum should be 1.0
        assert np.all(predictions >= 0)  # All values should be non-negative
        
    def test_format_counts_to_distribution(self):
        """Test conversion from counts dict to distribution array."""
        # Test for 2 qubits
        counts = {'00': 40, '01': 30, '10': 20, '11': 10}
        distribution = QuantumErrorNeuralNet.format_counts_to_distribution(counts, num_qubits=2)
        
        expected = np.array([[0.4, 0.3, 0.2, 0.1]])  # Reshape to (1, 4)
        assert distribution.shape == (1, 4)
        assert np.allclose(distribution, expected)
        
        # Test with missing bitstrings (should be filled with zeros)
        counts = {'00': 70, '11': 30}  # Missing '01' and '10'
        distribution = QuantumErrorNeuralNet.format_counts_to_distribution(counts, num_qubits=2)
        
        expected = np.array([[0.7, 0.0, 0.0, 0.3]])  # Reshape to (1, 4)
        assert distribution.shape == (1, 4)
        assert np.allclose(distribution, expected)
        
    def test_distribution_to_counts(self):
        """Test conversion from distribution array to counts dict."""
        # Test for 2 qubits
        distribution = np.array([0.4, 0.3, 0.2, 0.1])
        counts = QuantumErrorNeuralNet.distribution_to_counts(distribution, total_shots=100)
        
        assert len(counts) <= 4  # Should have at most 4 entries for 2 qubits
        assert sum(counts.values()) == 100  # Total shots should be preserved
        
        # Check that the distribution is roughly preserved
        # This is probabilistic, so we can't check exact values
        assert counts.get('00', 0) >= 35  # Roughly 40%, allowing some rounding
        assert counts.get('01', 0) >= 25  # Roughly 30%
        assert counts.get('10', 0) >= 15  # Roughly 20%
        assert counts.get('11', 0) >= 5   # Roughly 10%
        
        # Test with a 2D array
        distribution = np.array([[0.25, 0.25, 0.25, 0.25]])
        counts = QuantumErrorNeuralNet.distribution_to_counts(distribution, total_shots=100)
        
        assert len(counts) <= 4
        assert sum(counts.values()) == 100
        for bitstring in ['00', '01', '10', '11']:
            assert counts.get(bitstring, 0) > 0  # All should have some counts
        
    def test_save_and_load(self):
        """Test saving and loading the model."""
        # Create and train a simple model
        np.random.seed(42)
        model = QuantumErrorNeuralNet(input_size=4, hidden_layer_sizes=(10,), random_state=42)
        
        # Generate some data
        true_distributions = np.array([[0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1]])
        noise = np.random.normal(0, 0.05, true_distributions.shape)
        noisy_distributions = np.maximum(true_distributions + noise, 0)
        row_sums = noisy_distributions.sum(axis=1, keepdims=True)
        noisy_distributions = noisy_distributions / row_sums
        
        # Train the model
        model.train(noisy_distributions, true_distributions)
        
        # Get predictions before saving
        original_predictions = model.predict(noisy_distributions)
        
        # Save and load the model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp:
            model_path = tmp.name
            
        try:
            # Save model
            model.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            loaded_model = QuantumErrorNeuralNet.load_model(model_path)
            
            # Check type
            assert isinstance(loaded_model, QuantumErrorNeuralNet)
            
            # Check predictions are the same
            loaded_predictions = loaded_model.predict(noisy_distributions)
            assert np.allclose(original_predictions, loaded_predictions)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_roundtrip_conversion(self):
        """Test roundtrip conversion: counts → distribution → counts."""
        original_counts = {'00': 500, '01': 300, '10': 150, '11': 50}
        total_shots = sum(original_counts.values())
        
        # Convert to distribution
        distribution = QuantumErrorNeuralNet.format_counts_to_distribution(
            original_counts, num_qubits=2)
        
        # Convert back to counts
        reconstructed_counts = QuantumErrorNeuralNet.distribution_to_counts(
            distribution, total_shots=total_shots)
        
        # Check that counts are roughly preserved
        # Due to rounding, they won't be identical
        assert sum(reconstructed_counts.values()) == total_shots
        for bitstring in original_counts:
            assert abs(reconstructed_counts.get(bitstring, 0) - original_counts[bitstring]) <= 1

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])