OPENQASM 3.0;

qubit q[1];
bit c[1];

h q[0];
measure q[0] -> c[0];