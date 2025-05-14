from qiskit import QuantumCircuit
from qem_ml.functions import end_to_end_error_mitigation
from qem_ml.run_model import run_model_with_inputs
import os


TEST_NAME = "half_adder_random"

# make the output directory relative to that of this specific file no matter where it is run
OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "test_results", f"{TEST_NAME}_results")

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# implementation https://qmunity.thequantuminsider.com/2024/06/11/building-a-half-adder-circuit/
circuit = QuantumCircuit(4, 2)

# initialization random from training
circuit.cx(0, 2)  # First XOR gate
circuit.cx(1, 2)  # Second XOR gate
circuit.ccx(0, 1, 3)  # AND gate for carry bit

# Measure the outputs
circuit.measure(2, 0)  # extract XOR value (sum bit)
circuit.measure(3, 1)  # extract AND value (carry bit)

# Run error mitigation with randomized inputs
model, results = end_to_end_error_mitigation(
    circuit=circuit,
    output_dir=OUTPUT_DIRECTORY,
    model_name=TEST_NAME,
    error_range=(0.01, 0.1),
    randomize_inputs=True,
    num_training_samples=10,
    verbose=False
)
assert(model is not None)
assert(results is not None)

# Set input values for testing
a = 1
b = 0

MODEL_PATH = os.path.join(OUTPUT_DIRECTORY, f"{TEST_NAME}.joblib")
print(f"Model saved to: {MODEL_PATH}")
assert os.path.exists(MODEL_PATH), f"Model file {MODEL_PATH} does not exist."

# Create half adder circuit with specific input values
circuit = QuantumCircuit(4, 2)

# Set input qubits based on a and b values
if a == 1:
    circuit.x(0)  # Set first input
if b == 1:
    circuit.x(1)  # Set second input

# Core half adder logic remains the same
circuit.cx(0, 2)  # First XOR gate
circuit.cx(1, 2)  # Second XOR gate
circuit.ccx(0, 1, 3)  # AND gate for carry bit

# Measure the outputs
circuit.measure(2, 0)  # extract XOR value (sum bit)
circuit.measure(3, 1)  # extract AND value (carry bit)

# Run test with the specified inputs
results = run_model_with_inputs(
    circuit_file=None,  # No file needed as we are passing the circuit directly
    circuit=circuit,    # Pass the circuit we created
    model_path=MODEL_PATH,
    input_values=[a, b, 0, 0],  # Not needed as we already set the inputs in the circuit
    error_rate=0.05,
    output_dir=OUTPUT_DIRECTORY,
)

assert(results is not None)