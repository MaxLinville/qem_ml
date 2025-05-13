from qiskit import QuantumCircuit
from qem_ml.circuits.simulator import Simulator
from qem_ml.models.qe_neural_net import QuantumErrorNeuralNet
import numpy as np
import matplotlib.pyplot as plt

# Create half adder circuit with specific inputs a=0, b=1
circuit = QuantumCircuit(4, 2)

# Set input qubits
# No need to apply X to qubit 0 since a=0
circuit.x(1)  # Set second input to 1

# Half adder logic
circuit.cx(0, 2)  # First XOR gate
circuit.cx(1, 2)  # Second XOR gate
circuit.ccx(0, 1, 3)  # AND gate for carry bit

# Measure the outputs
circuit.measure(2, 0)  # extract XOR value (sum bit)
circuit.measure(3, 1)  # extract AND value (carry bit)

# Save circuit diagram
circuit.draw(output='mpl', filename='half_adder_debug_circuit.png')

print("Expected classical result for inputs a=0, b=1:")
print(f"  sum = {0 ^ 1} (qubit 2)")
print(f"  carry = {0 & 1} (qubit 3)")

# Create simulator with this circuit
simulator = Simulator(circuit)

# Get clean distribution (ground truth) with detailed debugging
print("\nRunning clean simulation...")
clean_counts = simulator.run_clean(generate_histogram=True, filename_prefix="debug_clean_")

print(f"\nRaw clean counts: {clean_counts}")

# Manually fix bitstrings
num_qubits = circuit.num_qubits
clean_counts_fixed = {}
for bitstring, count in clean_counts.items():
    print(f"  Original bitstring: '{bitstring}' with count: {count}")
    fixed_bitstring = bitstring.replace(" ", "")
    print(f"  After space removal: '{fixed_bitstring}'")
    
    if len(fixed_bitstring) > num_qubits:
        fixed_bitstring = fixed_bitstring[-num_qubits:]
        print(f"  Truncated (too long): '{fixed_bitstring}'")
    elif len(fixed_bitstring) < num_qubits:
        fixed_bitstring = fixed_bitstring.zfill(num_qubits)
        print(f"  Zero-padded (too short): '{fixed_bitstring}'")
        
    clean_counts_fixed[fixed_bitstring] = count
    print(f"  Final fixed bitstring: '{fixed_bitstring}' with count: {count}")

print(f"\nFixed clean counts: {clean_counts_fixed}")

# Convert to distribution using the class method
ideal_dist = QuantumErrorNeuralNet.format_counts_to_distribution(
    clean_counts_fixed, num_qubits=num_qubits
)

# Detailed inspection of the distribution
print("\nIdeal distribution inspection:")
print(f"  Shape: {ideal_dist.shape}")
print(f"  Sum: {np.sum(ideal_dist)}")
print(f"  Non-zero indices: {np.nonzero(ideal_dist)[1]}")
print(f"  Non-zero values: {ideal_dist[0, np.nonzero(ideal_dist)[1]]}")

# Visualize the ideal distribution
plt.figure(figsize=(10, 6))
plt.bar(range(len(ideal_dist[0])), ideal_dist[0])
plt.xlabel('State index')
plt.ylabel('Probability')
plt.title('Ideal Distribution for Half Adder (a=0, b=1)')
plt.xticks(range(len(ideal_dist[0])), 
           [format(i, f'0{num_qubits}b') for i in range(len(ideal_dist[0]))],
           rotation=90)
plt.tight_layout()
plt.savefig('debug_ideal_distribution.png')

print("\nDebug visualization saved as 'debug_ideal_distribution.png'")