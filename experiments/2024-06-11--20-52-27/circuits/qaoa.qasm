OPENQASM 2.0;
include "qelib1.inc";
gate exp_it_IIIIIZIZ_ q0,q1,q2,q3,q4,q5,q6,q7 { rzz(3.7974765914454456) q0,q2; }
gate exp_it_IZIIIIIZ_ q0,q1,q2,q3,q4,q5,q6,q7 { rzz(3.7974765914454456) q0,q6; }
gate exp_it_IIZIIIZI_ q0,q1,q2,q3,q4,q5,q6,q7 { rzz(3.7974765914454456) q1,q5; }
gate exp_it_ZIIIIIZI_ q0,q1,q2,q3,q4,q5,q6,q7 { rzz(3.7974765914454456) q1,q7; }
gate exp_it_IIIIZZII_ q0,q1,q2,q3,q4,q5,q6,q7 { rzz(3.7974765914454456) q2,q3; }
gate exp_it_ZIIIZIII_ q0,q1,q2,q3,q4,q5,q6,q7 { rzz(3.7974765914454456) q3,q7; }
gate exp_it_IIZZIIII_ q0,q1,q2,q3,q4,q5,q6,q7 { rzz(3.7974765914454456) q4,q5; }
gate exp_it_IZIZIIII_ q0,q1,q2,q3,q4,q5,q6,q7 { rzz(3.7974765914454456) q4,q6; }
gate gate_PauliEvolution(param0) q0,q1,q2,q3,q4,q5,q6,q7 { exp_it_IIIIIZIZ_ q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IZIIIIIZ_ q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIZIIIZI_ q0,q1,q2,q3,q4,q5,q6,q7; exp_it_ZIIIIIZI_ q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIIIZZII_ q0,q1,q2,q3,q4,q5,q6,q7; exp_it_ZIIIZIII_ q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIZZIIII_ q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IZIZIIII_ q0,q1,q2,q3,q4,q5,q6,q7; }
gate exp_it_XIIIIIII_ q0,q1,q2,q3,q4,q5,q6,q7 { rx(11.323468177675958) q7; }
gate exp_it_IXIIIIII_ q0,q1,q2,q3,q4,q5,q6,q7 { rx(11.323468177675958) q6; }
gate exp_it_IIXIIIII_ q0,q1,q2,q3,q4,q5,q6,q7 { rx(11.323468177675958) q5; }
gate exp_it_IIIXIIII_ q0,q1,q2,q3,q4,q5,q6,q7 { rx(11.323468177675958) q4; }
gate exp_it_IIIIXIII_ q0,q1,q2,q3,q4,q5,q6,q7 { rx(11.323468177675958) q3; }
gate exp_it_IIIIIXII_ q0,q1,q2,q3,q4,q5,q6,q7 { rx(11.323468177675958) q2; }
gate exp_it_IIIIIIXI_ q0,q1,q2,q3,q4,q5,q6,q7 { rx(11.323468177675958) q1; }
gate exp_it_IIIIIIIX_ q0,q1,q2,q3,q4,q5,q6,q7 { rx(11.323468177675958) q0; }
gate gate_PauliEvolution_139634390195152(param0) q0,q1,q2,q3,q4,q5,q6,q7 { exp_it_XIIIIIII_ q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IXIIIIII_ q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIXIIIII_ q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIIXIIII_ q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIIIXIII_ q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIIIIXII_ q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIIIIIXI_ q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIIIIIIX_ q0,q1,q2,q3,q4,q5,q6,q7; }
gate exp_it_IIIIIZIZ__139634330508144 q0,q1,q2,q3,q4,q5,q6,q7 { rzz(1.2429950269704841) q0,q2; }
gate exp_it_IZIIIIIZ__139634227190304 q0,q1,q2,q3,q4,q5,q6,q7 { rzz(1.2429950269704841) q0,q6; }
gate exp_it_IIZIIIZI__139634330511264 q0,q1,q2,q3,q4,q5,q6,q7 { rzz(1.2429950269704841) q1,q5; }
gate exp_it_ZIIIIIZI__139634295904992 q0,q1,q2,q3,q4,q5,q6,q7 { rzz(1.2429950269704841) q1,q7; }
gate exp_it_IIIIZZII__139634295909072 q0,q1,q2,q3,q4,q5,q6,q7 { rzz(1.2429950269704841) q2,q3; }
gate exp_it_ZIIIZIII__139634352220832 q0,q1,q2,q3,q4,q5,q6,q7 { rzz(1.2429950269704841) q3,q7; }
gate exp_it_IIZZIIII__139634352212384 q0,q1,q2,q3,q4,q5,q6,q7 { rzz(1.2429950269704841) q4,q5; }
gate exp_it_IZIZIIII__139634526069536 q0,q1,q2,q3,q4,q5,q6,q7 { rzz(1.2429950269704841) q4,q6; }
gate gate_PauliEvolution_139634310668656(param0) q0,q1,q2,q3,q4,q5,q6,q7 { exp_it_IIIIIZIZ__139634330508144 q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IZIIIIIZ__139634227190304 q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIZIIIZI__139634330511264 q0,q1,q2,q3,q4,q5,q6,q7; exp_it_ZIIIIIZI__139634295904992 q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIIIZZII__139634295909072 q0,q1,q2,q3,q4,q5,q6,q7; exp_it_ZIIIZIII__139634352220832 q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIZZIIII__139634352212384 q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IZIZIIII__139634526069536 q0,q1,q2,q3,q4,q5,q6,q7; }
gate exp_it_XIIIIIII__139633957054416 q0,q1,q2,q3,q4,q5,q6,q7 { rx(-0.6561130500198156) q7; }
gate exp_it_IXIIIIII__139633991398400 q0,q1,q2,q3,q4,q5,q6,q7 { rx(-0.6561130500198156) q6; }
gate exp_it_IIXIIIII__139634578117472 q0,q1,q2,q3,q4,q5,q6,q7 { rx(-0.6561130500198156) q5; }
gate exp_it_IIIXIIII__139634330515104 q0,q1,q2,q3,q4,q5,q6,q7 { rx(-0.6561130500198156) q4; }
gate exp_it_IIIIXIII__139634225453552 q0,q1,q2,q3,q4,q5,q6,q7 { rx(-0.6561130500198156) q3; }
gate exp_it_IIIIIXII__139634225454224 q0,q1,q2,q3,q4,q5,q6,q7 { rx(-0.6561130500198156) q2; }
gate exp_it_IIIIIIXI__139634225456528 q0,q1,q2,q3,q4,q5,q6,q7 { rx(-0.6561130500198156) q1; }
gate exp_it_IIIIIIIX__139634225450720 q0,q1,q2,q3,q4,q5,q6,q7 { rx(-0.6561130500198156) q0; }
gate gate_PauliEvolution_139634295913200(param0) q0,q1,q2,q3,q4,q5,q6,q7 { exp_it_XIIIIIII__139633957054416 q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IXIIIIII__139633991398400 q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIXIIIII__139634578117472 q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIIXIIII__139634330515104 q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIIIXIII__139634225453552 q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIIIIXII__139634225454224 q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIIIIIXI__139634225456528 q0,q1,q2,q3,q4,q5,q6,q7; exp_it_IIIIIIIX__139634225450720 q0,q1,q2,q3,q4,q5,q6,q7; }
qreg q[8];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
gate_PauliEvolution(3.7974765914454456) q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
gate_PauliEvolution_139634390195152(5.661734088837979) q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
gate_PauliEvolution_139634310668656(1.2429950269704841) q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
gate_PauliEvolution_139634295913200(-0.3280565250099078) q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
