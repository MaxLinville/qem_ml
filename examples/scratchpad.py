from qiskit import QuantumCircuit
from qem_ml.functions import end_to_end_error_mitigation
from qem_ml.circuits import load_circuit_from_file

# Create or load your circuit
# circuit = QuantumCircuit(3)
# circuit.h(0)
# circuit.cx(0, 1)
# circuit.cx(0, 2)
# circuit.measure_all()

circuit = load_circuit_from_file("test_circuits/qft.qasm")

# Run error mitigation
model, results = end_to_end_error_mitigation(
    circuit=circuit,
    output_dir="./mitigation_results",
    model_name="ghz_state"
)

print(f"Model training score: {results['training_score']}")
print(f"Test MSE: {results.get('test_mse', 'N/A')}")