from qiskit import QuantumCircuit
from qem_ml.functions import end_to_end_error_mitigation
from qem_ml.circuits import load_circuit_from_file


TEST_NAME = "qft"
OUTPUT_DIRECTORY = f"./test_results/{TEST_NAME}_results"
INPUT_FILE = f"test_circuits/qft.qasm"

circuit: QuantumCircuit = load_circuit_from_file(INPUT_FILE)
circuit.draw(output='mpl', filename=f"{OUTPUT_DIRECTORY}/circuit_drawing.png")