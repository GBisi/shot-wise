OPENQASM 2.0;
include "qelib1.inc";
gate exp_it_IIIZZ_ q0,q1,q2,q3,q4 { rzz(4.353113097887503) q0,q1; }
gate exp_it_IIZIZ_ q0,q1,q2,q3,q4 { rzz(4.353113097887503) q0,q2; }
gate exp_it_IZIZI_ q0,q1,q2,q3,q4 { rzz(4.353113097887503) q1,q3; }
gate exp_it_ZIZII_ q0,q1,q2,q3,q4 { rzz(4.353113097887503) q2,q4; }
gate exp_it_ZZIII_ q0,q1,q2,q3,q4 { rzz(4.353113097887503) q3,q4; }
gate gate_PauliEvolution(param0) q0,q1,q2,q3,q4 { exp_it_IIIZZ_ q0,q1,q2,q3,q4; exp_it_IIZIZ_ q0,q1,q2,q3,q4; exp_it_IZIZI_ q0,q1,q2,q3,q4; exp_it_ZIZII_ q0,q1,q2,q3,q4; exp_it_ZZIII_ q0,q1,q2,q3,q4; }
gate exp_it_XIIII_ q0,q1,q2,q3,q4 { rx(-12.06859726452274) q4; }
gate exp_it_IXIII_ q0,q1,q2,q3,q4 { rx(-12.06859726452274) q3; }
gate exp_it_IIXII_ q0,q1,q2,q3,q4 { rx(-12.06859726452274) q2; }
gate exp_it_IIIXI_ q0,q1,q2,q3,q4 { rx(-12.06859726452274) q1; }
gate exp_it_IIIIX_ q0,q1,q2,q3,q4 { rx(-12.06859726452274) q0; }
gate gate_PauliEvolution_139906099769216(param0) q0,q1,q2,q3,q4 { exp_it_XIIII_ q0,q1,q2,q3,q4; exp_it_IXIII_ q0,q1,q2,q3,q4; exp_it_IIXII_ q0,q1,q2,q3,q4; exp_it_IIIXI_ q0,q1,q2,q3,q4; exp_it_IIIIX_ q0,q1,q2,q3,q4; }
gate exp_it_IIIZZ__139906099768064 q0,q1,q2,q3,q4 { rzz(-3.6398398371764604) q0,q1; }
gate exp_it_IIZIZ__139906099770464 q0,q1,q2,q3,q4 { rzz(-3.6398398371764604) q0,q2; }
gate exp_it_IZIZI__139906099761344 q0,q1,q2,q3,q4 { rzz(-3.6398398371764604) q1,q3; }
gate exp_it_ZIZII__139906100330352 q0,q1,q2,q3,q4 { rzz(-3.6398398371764604) q2,q4; }
gate exp_it_ZZIII__139906100329872 q0,q1,q2,q3,q4 { rzz(-3.6398398371764604) q3,q4; }
gate gate_PauliEvolution_139906100328576(param0) q0,q1,q2,q3,q4 { exp_it_IIIZZ__139906099768064 q0,q1,q2,q3,q4; exp_it_IIZIZ__139906099770464 q0,q1,q2,q3,q4; exp_it_IZIZI__139906099761344 q0,q1,q2,q3,q4; exp_it_ZIZII__139906100330352 q0,q1,q2,q3,q4; exp_it_ZZIII__139906100329872 q0,q1,q2,q3,q4; }
gate exp_it_XIIII__139906100565360 q0,q1,q2,q3,q4 { rx(1.9303304804264925) q4; }
gate exp_it_IXIII__139906100570688 q0,q1,q2,q3,q4 { rx(1.9303304804264925) q3; }
gate exp_it_IIXII__139906100563584 q0,q1,q2,q3,q4 { rx(1.9303304804264925) q2; }
gate exp_it_IIIXI__139906099761104 q0,q1,q2,q3,q4 { rx(1.9303304804264925) q1; }
gate exp_it_IIIIX__139906099764752 q0,q1,q2,q3,q4 { rx(1.9303304804264925) q0; }
gate gate_PauliEvolution_139906099764560(param0) q0,q1,q2,q3,q4 { exp_it_XIIII__139906100565360 q0,q1,q2,q3,q4; exp_it_IXIII__139906100570688 q0,q1,q2,q3,q4; exp_it_IIXII__139906100563584 q0,q1,q2,q3,q4; exp_it_IIIXI__139906099761104 q0,q1,q2,q3,q4; exp_it_IIIIX__139906099764752 q0,q1,q2,q3,q4; }
qreg q[5];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
gate_PauliEvolution(4.353113097887503) q[0],q[1],q[2],q[3],q[4];
gate_PauliEvolution_139906099769216(-6.03429863226137) q[0],q[1],q[2],q[3],q[4];
gate_PauliEvolution_139906100328576(-3.6398398371764604) q[0],q[1],q[2],q[3],q[4];
gate_PauliEvolution_139906099764560(0.9651652402132462) q[0],q[1],q[2],q[3],q[4];
