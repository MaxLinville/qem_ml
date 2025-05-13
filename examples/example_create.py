from qiskit import QuantumCircuit
from qem_ml.functions import end_to_end_error_mitigation

TEST_NAME = "ghz"
OUTPUT_DIRECTORY = f"./test_results/{TEST_NAME}_results"

circuit: QuantumCircuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(0, 2)
circuit.measure_all()
circuit.draw(output='mpl', filename=f"{OUTPUT_DIRECTORY}/circuit_drawing.png")

# Run error mitigation
model, results = end_to_end_error_mitigation(
    circuit=circuit,
    output_dir=OUTPUT_DIRECTORY,
    model_name=TEST_NAME,
)

print(f"Model training score: {results['training_score']}")
print(f"Test MSE: {results.get('test_mse', 'N/A')}")