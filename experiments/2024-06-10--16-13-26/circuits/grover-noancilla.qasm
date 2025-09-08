OPENQASM 2.0;
include "qelib1.inc";
gate mcphase(param0) q0,q1,q2,q3,q4 { cp(pi/8) q3,q4; cx q3,q2; cp(-pi/8) q2,q4; cx q3,q2; cp(pi/8) q2,q4; cx q2,q1; cp(-pi/8) q1,q4; cx q3,q1; cp(pi/8) q1,q4; cx q2,q1; cp(-pi/8) q1,q4; cx q3,q1; cp(pi/8) q1,q4; cx q1,q0; cp(-pi/8) q0,q4; cx q3,q0; cp(pi/8) q0,q4; cx q2,q0; cp(-pi/8) q0,q4; cx q3,q0; cp(pi/8) q0,q4; cx q1,q0; cp(-pi/8) q0,q4; cx q3,q0; cp(pi/8) q0,q4; cx q2,q0; cp(-pi/8) q0,q4; cx q3,q0; cp(pi/8) q0,q4; }
gate mcx q0,q1,q2,q3 { h q3; p(pi/8) q0; p(pi/8) q1; p(pi/8) q2; p(pi/8) q3; cx q0,q1; p(-pi/8) q1; cx q0,q1; cx q1,q2; p(-pi/8) q2; cx q0,q2; p(pi/8) q2; cx q1,q2; p(-pi/8) q2; cx q0,q2; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; h q3; }
gate gate_Q q0,q1,q2,q3,q4 { mcphase(pi) q0,q1,q2,q3,q4; h q3; h q2; h q1; h q0; x q0; x q1; x q2; x q3; h q3; mcx q0,q1,q2,q3; h q3; x q0; x q1; x q2; x q3; h q0; h q1; h q2; h q3; }
gate gate_Q_139906100576160 q0,q1,q2,q3,q4 { gate_Q q0,q1,q2,q3,q4; }
gate gate_Q_139906100571120 q0,q1,q2,q3,q4 { gate_Q q0,q1,q2,q3,q4; }
gate gate_Q_139906099915520 q0,q1,q2,q3,q4 { gate_Q q0,q1,q2,q3,q4; }
qreg q[4];
qreg flag[1];
h q[0];
h q[1];
h q[2];
h q[3];
x flag[0];
gate_Q_139906100576160 q[0],q[1],q[2],q[3],flag[0];
gate_Q_139906100571120 q[0],q[1],q[2],q[3],flag[0];
gate_Q_139906099915520 q[0],q[1],q[2],q[3],flag[0];
