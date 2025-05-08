// This creates a GHZ state preparation circuit
OPENQASM 3.0;
include "stdgates.inc";

// Define 3 qubits
qubit[3] q;

// Create GHZ state
h q[0];
cx q[0], q[1];
cx q[0], q[2];

// Measure all qubits
bit[3] c;
c = measure q;