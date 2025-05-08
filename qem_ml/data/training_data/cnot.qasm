OPENQASM 3.0;

qubit q[2];
bit c[2];

cx q[0], q[1];

measure q[0] -> c[0];
measure q[1] -> c[1];