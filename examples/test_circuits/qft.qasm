// This implements a 5-qubit Quantum Fourier Transform followed by inverse QFT
OPENQASM 3.0;
include "stdgates.inc";

// Define 5 qubits
qubit[5] q;

// QFT implementation
// Apply Hadamard to all qubits
h q[0];
// Controlled rotations for q[0]
cp(pi/2) q[1], q[0];
cp(pi/4) q[2], q[0];
cp(pi/8) q[3], q[0];
cp(pi/16) q[4], q[0];

// Second qubit transformations
h q[1];
// Controlled rotations for q[1]
cp(pi/2) q[2], q[1];
cp(pi/4) q[3], q[1];
cp(pi/8) q[4], q[1];

// Third qubit transformations
h q[2];
// Controlled rotations for q[2]
cp(pi/2) q[3], q[2];
cp(pi/4) q[4], q[2];

// Fourth qubit transformations
h q[3];
// Controlled rotation for q[3]
cp(pi/2) q[4], q[3];

// Final Hadamard
h q[4];

// Barrier to separate QFT and inverse QFT
barrier q;

// Inverse QFT implementation
// Starts with the last qubit
h q[4];

// Inverse rotations for q[3]
cp(-pi/2) q[4], q[3];
h q[3];

// Inverse rotations for q[2]
cp(-pi/4) q[4], q[2];
cp(-pi/2) q[3], q[2];
h q[2];

// Inverse rotations for q[1]
cp(-pi/8) q[4], q[1];
cp(-pi/4) q[3], q[1];
cp(-pi/2) q[2], q[1];
h q[1];

// Inverse rotations for q[0]
cp(-pi/16) q[4], q[0];
cp(-pi/8) q[3], q[0];
cp(-pi/4) q[2], q[0];
cp(-pi/2) q[1], q[0];
h q[0];

// Measure all qubits
bit[5] c;
c = measure q;