OPENQASM 3.0;

qubit q[3];
bit c[3];

ccx q[0], q[1], q[2];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];