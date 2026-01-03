"""
Copyright 2024 Shot-wise contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import json
import math
import shutil
import random
import datetime


from tqdm import tqdm
from itertools import chain, combinations

import numpy as np

from mqt.bench import get_benchmark

import qiskit_aer.noise as noise
from qiskit_ibm_provider import IBMProvider
from qiskit.quantum_info import random_unitary
from qiskit import QuantumCircuit
from qiskit.extensions import UnitaryGate

from qukit.components.backends.ionq import IonQBackend

from qukit.components import VirtualProvider, Translator, Dispatcher, CompilationManager
from qukit.components.dispatcher import Dispatch

from qukit.components.backends import AerStateBackend, AerBackend

import configparser

circuits_to_remove = []

def debug(*args, **kwargs):
    print(f"[{datetime.datetime.now().strftime('%d-%m-%Y--%H-%M-%S')}]", *args, **kwargs)

CONFIG = configparser.ConfigParser()
CONFIG.optionxform=str # Preserve case
CONFIG.read("config.ini")

TOKENS: dict[str, str | bool] = {k:v for k,v in CONFIG["TOKENS"].items()}
for v in TOKENS:
    if v.lower() == "none":
        TOKENS[v] = None
    elif v == "true":
        TOKENS[v] = True
    elif v == "false":
        TOKENS[v] = False

INITIAL_PLACEMENT = {
    "ibm_kyoto": {
        4: [8, 16, 25, 26],
        5: [8, 9, 16, 25, 26],
        6: [8, 9, 10, 16, 25, 26],
        8: [8,  9, 10, 11, 12, 16, 25, 26]
    },
    "ibm_brisbane": {
        4: [32, 36, 50, 51],
        5: [31, 32, 36, 50, 51],
        6: [17, 30, 31, 32, 36, 51],
        8: [11, 12, 17, 30, 31, 32, 36, 51]
    },
    "ibm_osaka": {
        4: [45, 46, 47, 54],
        5: [35, 45, 46, 47, 54],
        6: [28, 35, 45, 46, 47, 54],
        8: [27, 28, 35, 45, 46, 47, 48, 54]
    },
}

def fair_policy(backends, calibration_report):
    return {b: 1/len(backends) for b in backends.keys()}

def random_policy(backends, calibration_report): 
    c = {b: 0 for b in backends.keys()}
    max = 1
    for b in backends.keys():
        r = random.uniform(0, max)
        c[b] += r
        max -= r
    
    c[list(backends.keys())[-1]] += max
    
    return c

def hellinger_policy(backends, calibration_report):
    reliability = calibration_report["reliability"]
    _reliability = {}
    for b_name, b_device in backends.items():
        if b_device in reliability:
            _reliability[b_name] = reliability[b_device]
        else:
            _reliability[b_name] = 0
    return {k: v/sum(_reliability.values()) for k,v in _reliability.items()}

def mise_policy(backends, calibration_report):
    w_bar = calibration_report["mise_weights"]
    key = "/".join([b for b in backends.values()])
    if key in w_bar:
        _w_bar = {}
        for b_name, b_device in backends.items():
            if b_device in w_bar[key]:
                _w_bar[b_name] = w_bar[key][b_device]
            else:
                _w_bar[b_name] = 0
    else:
        all_equal = True
        for b in backends.values():
            if b != list(backends.values())[0]:
                all_equal = False
                break
        if not all_equal:
            debug(f"MISE weights not found for {key}")
        _w_bar = {b: 1/len(backends) for b in backends.keys()}
    
    return _w_bar

POLICIES = {
    "fair": fair_policy,
    "random": random_policy,
    "hellinger": hellinger_policy,
    "mise": mise_policy,
}


tr = None
vp = None
disp = None
cm = None
provider = None

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def counts_dict_to_array(counts, qubits):
    _counts = []
    for i in range(2**qubits):
        bin_i = bin(i)[2:]
        bin_i = "0"*(qubits-len(bin_i))+bin_i
        if bin_i in counts:
            _counts.append(counts[bin_i])
        else:
            _counts.append(0)
    return np.array(_counts)

def counts_dict_to_probs(counts, qubits):
    _counts = counts_dict_to_array(counts, qubits)
    shots = sum(_counts)
    return np.array(_counts)/shots

def Hellinger_exact_exact(dis1, dis2):
    arg = 1-np.sum(np.sqrt(dis1*dis2))
    return np.sqrt(max(arg, 0))
def Hellinger_dist1_exact_naive(counts, dis_exact):
    nshots=np.sum(counts)
    arg = 1-np.sum(np.sqrt((counts/nshots)*dis_exact))
    return np.sqrt(max(arg, 0))
def Hellinger_dist1_exact_jack(counts, dis_exact):
    D=len(dis_exact)
    nshots=np.sum(counts) 
    H_naive=Hellinger_dist1_exact_naive(counts, dis_exact)
    H_jack_samples = np.array([(0 if counts[y]==0 else np.sqrt(1-1./np.sqrt(nshots-1)*( np.sqrt(nshots)*(1-H_naive**2)-(np.sqrt(counts[y])-np.sqrt(counts[y]-1))*np.sqrt(dis_exact[y]) ))) for y in range(D)])
    H_jack = 1./nshots*np.sum([counts[y]*H_jack_samples[y] for y in range(D)])
    dH_jack=np.sqrt((nshots-1)/nshots*np.sum([counts[y]*(H_jack_samples[y]-H_jack)**2 for y in range(D)]))
    H_unbiased = H_jack+nshots*(H_naive-H_jack)
    return H_unbiased,dH_jack


def hellinger_distance(counts, ground_truth, qubits):
    _counts = []
    _ground_truth = []
    for i in range(2**qubits):
        bin_i = bin(i)[2:]
        bin_i = "0"*(qubits-len(bin_i))+bin_i
        if bin_i in counts:
            _counts.append(counts[bin_i])
        else:
            _counts.append(0)
            
        if bin_i in ground_truth:
            _ground_truth.append(ground_truth[bin_i])
        else:
            _ground_truth.append(0)
            
    counts = np.array(_counts)
    ground_truth = np.array(_ground_truth)

    return Hellinger_dist1_exact_jack(counts, ground_truth)


def hellinger_distance_exact_exact(dist1, dist2, qubits):
    _dist1 = []
    _dist2 = []
    for i in range(2**qubits):
        bin_i = bin(i)[2:]
        bin_i = "0"*(qubits-len(bin_i))+bin_i
        if bin_i in dist1:
            _dist1.append(dist1[bin_i])
        else:
            _dist1.append(0)
            
        if bin_i in dist2:
            _dist2.append(dist2[bin_i])
        else:
            _dist2.append(0)
            
    dist1 = np.array(_dist1)
    dist2 = np.array(_dist2)
    
    return Hellinger_exact_exact(dist1, dist2)

def compute_reliability(backend_name, distances, eps_rel=1e-10, gamma=1.0):
    distances = [d[0] for d in distances]
    avg_distance =  sum(distances)/len(distances)
    return (avg_distance+eps_rel)**(-gamma)

def compute_mise_weights(backends, circuits, ground_truth, counts, qubits):
    Cmat = np.zeros((len(backends), len(backends)))
    fvec = np.zeros(len(backends))
    for c_i,c in enumerate(circuits):
        p_exact_realcase = counts_dict_to_probs(ground_truth[c_i], qubits)
        p_biases_realcase = {}
        for b in backends:
            probs = [counts_dict_to_probs(_counts, qubits) for _counts in counts[b][c_i]]
            p_biases_realcase[b] = np.mean(probs, axis=0)
            
        for i in range(len(backends)):
            Cmat[i,i] += np.sum(p_biases_realcase[backends[i]]**2)
            for j in range(i+1, len(backends)):
                s = np.sum(p_biases_realcase[backends[i]]*p_biases_realcase[backends[j]])
                Cmat[i,j] += s
                Cmat[j,i] += s
                
            fvec[i] += np.sum(p_exact_realcase*p_biases_realcase[backends[i]])
            
    Cmat = Cmat/len(circuits)
    fvec = fvec/len(circuits)
    
    Cmat_inv = np.linalg.inv(Cmat)
    mu_bar=(np.sum(Cmat_inv@fvec)-1)/np.sum(Cmat_inv)
    
    w_bar=Cmat_inv@(fvec-mu_bar)
    
    return w_bar

def get_noise_models() -> dict:
    noise_models = {}
    backends = [b for b in provider.backends() if "simulator" not in b.name]
    for b in backends:
        n_m = noise.NoiseModel.from_backend(b)
        noise_models[b.name] = n_m
    return noise_models

def get_properties() -> dict:
    properties = {}
    backends = [b for b in provider.backends() if "simulator" not in b.name]
    for b in backends:
        p = noise.NoiseModel.from_backend(b)
        properties[b.name] = p
    return properties

def get_backends(blueprint:dict, noise_models: dict) -> None:
    backends = []
    for k,v in blueprint.items():
        for i in range(v):
            if k.startswith("ibm"):
                backends.append(AerBackend(k, vp.get_provider("Local"), noise_model=noise_models[k], instance_name=f"{k}_{i}"))
            else:
                backends.append(IonQBackend("simulator", vp.get_provider("IonQ"), noise_model=k, instance_name=f"ionq_simulator_{k}_{i}"))
    return backends

def save_backends_info(folder):
    os.makedirs(folder+"/noise_models", exist_ok=True)
    os.makedirs(folder+"/properties", exist_ok=True)
    
    json.dump({b.name():b.device() for b in backends}, open(f"{folder}/backends.json", "w"), indent=4)
    
    for k,v in noise_models.items():
        json.dump(v.to_dict(serializable=True), open(f"{folder}/noise_models/{k}.json", "w"), indent=4)
            
            
    def datetime_converter(o):
        if isinstance(o, datetime.datetime):
            return o.__str__()
        
    for k,v in properties.items():
        json.dump(v.to_dict(), open(f"{folder}/properties/{k}.json", "w"), indent=4, default=datetime_converter)
        
        
def compute_ground_truth(circuits: QuantumCircuit):
    ground_truth = []
    vp = VirtualProvider({"Local": True})
    oracle = AerStateBackend("oracle", vp.get_provider("Local"))
    
    for c in tqdm(circuits):
        ground_truth.append((oracle.run(c.qasm())).distribution())
        
    return ground_truth
    

def collect_counts(circuits: QuantumCircuit, shots, backends, rounds=1, initial_placement: dict[str, list[int]] = {}):
    counts = {}

    for c in tqdm(circuits):
        c.measure_all()

        last_c = None
        while last_c is None or last_c != c.qasm():
            last_c = c.qasm()
            c = c.decompose()
        
        d = []
        _c = c.copy()
        for backend in backends:
            if backend.device() not in counts:
                counts[backend.device()] = [[]]
            else:
                counts[backend.device()].append([])
            if backend.device() in initial_placement:
                c = cm.compile(c, "qiskit", backend, optimisation_level=0, initial_placement=initial_placement[backend.device()])
                c = c.get()
            d.append((backend, c, shots))  
            c = _c         
            
        dispatch = Dispatch(d)
        for _ in tqdm(range(rounds)):
            results = disp.run(dispatch)

            for r in results:
                _counts = r.counts()
                counts[r.backend().device()][-1].append(_counts)
                
    return counts

def sample_counts(counts, shots):
    total_shots = sum(counts.values())
    p = [v/total_shots for v in counts.values()]
    c=np.random.multinomial(shots,p)

    c_tmp=np.copy(c)
    for _ in range(shots):
        idx=np.argmax(np.random.multinomial(1,c_tmp/np.sum(c_tmp)))
        c_tmp[idx]-=1
    
    return {k: v for k,v in zip(counts.keys(), c)}

def setup_calibration(now, existing_circuits = False):
    debug("Calibrating...")
    os.makedirs(f"calibrations/data/{now}", exist_ok=True)
    os.makedirs(f"calibrations/circuits", exist_ok=True)
    
    num_circuits = int(CONFIG["CALIBRATION"]["circuits"])
    num_existing_circuits = len([f for f in os.listdir("calibrations/circuits") if f.endswith(".qasm")])
    
    if existing_circuits:
        if num_existing_circuits < num_circuits:
            debug("Generating circuits...")
            states = 2**qubits
            for i in range(num_circuits - num_existing_circuits):
                qc = QuantumCircuit(qubits)
                haar_random_gate = UnitaryGate(random_unitary(states))
                qc.append(haar_random_gate, range(qubits))
                last_qc = None
                while last_qc is None or last_qc != qc.qasm():
                    last_qc = qc.qasm()
                    qc = qc.decompose()
                
                with open(f"calibrations/circuits/circuit_{i+num_existing_circuits}.qasm", "w") as f:
                    f.write(qc.qasm())
            debug(f"Generated {num_circuits - num_existing_circuits} circuits!")
        circuits = [QuantumCircuit.from_qasm_file(f"calibrations/circuits/{f}") for f in os.listdir("calibrations/circuits")]
        circuits = sorted(circuits, key=lambda x: len(x), reverse=True)
        circuits = circuits[:num_circuits]
    else:
        debug("Not using existing circuits!")
        debug("Generating circuits...")
        circuits = []
        states = 2**qubits
        for i in range(num_circuits):
            qc = QuantumCircuit(qubits)
            haar_random_gate = UnitaryGate(random_unitary(states))
            qc.append(haar_random_gate, range(qubits))
            last_qc = None
            while last_qc is None or last_qc != qc.qasm():
                last_qc = qc.qasm()
                qc = qc.decompose()
            
            with open(f"calibrations/circuits/circuit_{(num_existing_circuits*10)+i}.qasm", "w") as f:
                f.write(qc.qasm())
            circuits_to_remove.append(f"calibrations/circuits/circuit_{(num_existing_circuits*10)+i}.qasm")
            circuits.append(qc)
        debug(f"Generated {num_circuits} circuits!")
        
    
    debug(f"Using {len(circuits)} circuits!")
    
    return circuits

def get_calibration_counts(now, circuits, backends, initial_placement):
    debug("Collecting calibration counts...")
    rounds = int(CONFIG["CALIBRATION"]["rounds"])
    shots_multiplier = int(CONFIG["CALIBRATION"]["shots_multiplier"])
    shots = (2**qubits) * shots_multiplier
    os.makedirs(f"calibrations/data/{now}/counts", exist_ok=True)
    os.makedirs(f"calibrations/data/{now}/counts/docs", exist_ok=True)
    os.system(f"cp config.ini run.py calibrations/data/{now}/counts/docs")
    
    with open(f"calibrations/data/{now}/q{qubits}-r{rounds}-c{len(circuits)}-b{len(backends)}", "w") as f:
        pass
    
    os.makedirs(f"calibrations/data/{now}/counts/backends", exist_ok=True)
    save_backends_info(f"calibrations/data/{now}/counts/backends")
    
    os.makedirs(f"calibrations/data/{now}/counts/circuits", exist_ok=True)
    for i,c in enumerate(circuits):
        with open(f"calibrations/data/{now}/counts/circuits/circuit_{i}.qasm", "w") as f:
            f.write(c.qasm())
    
    counts = collect_counts(circuits, shots, backends, rounds, initial_placement)
    with open(f"calibrations/data/{now}/counts/counts.json", "w") as f:
        f.write(json.dumps(counts))
        
    return counts

def analyse_calibration_counts(now, counts, circuits):
    
    debug("Analysing calibration counts...")
    analysis_now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    os.makedirs(f"calibrations/data/{now}/analysis/{analysis_now}", exist_ok=True)
    os.makedirs(f"calibrations/data/{now}/analysis/{analysis_now}/docs", exist_ok=True)
    os.system(f"cp config.ini run.py calibrations/data/{now}/analysis/{analysis_now}/docs")
    
    qubits = len(list(counts[list(counts.keys())[0]][0][0].keys())[0])    
    backends = list(counts.keys())
    
    gt = compute_ground_truth(circuits)
    json.dump(gt, open(f"calibrations/data/{now}/analysis/{analysis_now}/ground_truth.json", "w"))
    
    distances = {}
    for backend, _counts in counts.items():
        distances[backend] = []
        for i in range(len(circuits)):
            for j in range(len(_counts[i])): #for each round
                distances[backend].append(hellinger_distance(_counts[i][j], gt[i], qubits))
            
    json.dump(distances, open(f"calibrations/data/{now}/analysis/{analysis_now}/distances.json", "w"), indent=4)
    
    eps_rel = float(CONFIG["CALIBRATION"]["eps_rel"])
    gamma = float(CONFIG["CALIBRATION"]["gamma"])
    
    reliability = {k: compute_reliability(k, v,  eps_rel=eps_rel, gamma=gamma) for k,v in distances.items()}
    json.dump(reliability, open(f"calibrations/data/{now}/analysis/{analysis_now}/reliability.json", "w"), indent=4)
    
    mise_weights = {}
    for bs in powerset(backends):
        if len(bs) == 0 or len(bs) == 1:
            continue
    
        valid_backends = ([b for b in bs]).copy()
        last_valid_backends = None
        w_bar = None
        while last_valid_backends is None or (valid_backends != last_valid_backends and valid_backends != []):
            last_valid_backends = valid_backends.copy()
            w_bar = compute_mise_weights(valid_backends, circuits, gt, counts, qubits)
            out_of_bounds=np.where((w_bar<0.))[0]
            valid_backends = [b for i,b in enumerate(valid_backends) if i not in out_of_bounds]
            
        w_bar = {b: w_bar[valid_backends.index(b)] if b in valid_backends else 0 for b in bs}
        mise_weights["/".join([b for b in bs])] = w_bar
    
    json.dump(mise_weights, open(f"calibrations/data/{now}/analysis/{analysis_now}/mise_weights.json", "w"), indent=4)
    
    calibration_report = {
        "calibration_counts_folder": now,
        "calibration_analysis_folder": analysis_now,
        "distances": distances,
        "reliability": reliability,
        "mise_weights": mise_weights,
        "circuits": [c.qasm() for c in circuits],
        "ground_truth": gt,
        "counts": counts,
    }
    
    json.dump(calibration_report, open(f"calibrations/data/{now}/analysis/{analysis_now}/calibration_report.json", "w"), indent=4, default=np_encoder)
    
    return calibration_report

def setup_experiment(now):
    debug("Setting up experiment...")
    os.makedirs(f"experiments/{now}", exist_ok=True)
    
    circuits = json.loads(CONFIG["EXPERIMENT"]["circuits"])
    _circuits = {}
    for c in circuits:
        qc = get_benchmark(c, "alg", qubits)
        qc.remove_final_measurements()
        _circuits[c] = qc
    circuits = _circuits
    os.makedirs(f"experiments/{now}/circuits", exist_ok=True)
    for k,v in circuits.items():
        with open(f"experiments/{now}/circuits/{k}.qasm", "w") as f:
            f.write(v.qasm())
            
    os.makedirs(f"experiments/{now}/docs", exist_ok=True)
    os.system(f"cp config.ini run.py experiments/{now}/docs")
    
    os.makedirs(f"experiments/{now}/backends", exist_ok=True)
    save_backends_info(f"experiments/{now}/backends")
    
    os.makedirs(f"experiments/{now}/baselines", exist_ok=True)
    os.makedirs(f"experiments/{now}/experiments", exist_ok=True)
        
    return circuits
    
def get_baselines(now, circuits, qubits):
    debug("Computing baselines...")
    shots = (2**qubits) * int(CONFIG["EXPERIMENT"]["shots_multiplier"])
    rounds = int(CONFIG["EXPERIMENT"]["rounds"])
    
    with open(f"experiments/{now}/q{qubits}-r{rounds}-c{len(circuits)}-b{len(backends)}", "w") as f:
        pass
    
    circuits = list(circuits.values())
    baselines = collect_counts(circuits, shots, backends, rounds, initial_placement)
    
    json.dump(baselines, open(f"experiments/{now}/baselines/baselines.json", "w"), indent=4)
            
    return baselines

def perform_experiments(bss, baselines, calibration_report, backend_to_device, policies, circuits):
    debug("Performing experiments...")
    rounds = len(list(baselines[list(baselines.keys())[0]][0]))
    shots = sum(baselines[list(baselines.keys())[0]][0][0].values())
    print("Rounds:", rounds)
    print("Shots:", shots)
    print("Policies:", policies)
    print("Circuits:", circuits)
    
    num_circuits = len(circuits)
    experiments = []
    
    for bs in tqdm(bss):
        if len(bs) == 0 or len(bs) == 1:
            continue
        for c_i in range(num_circuits):
            for r_i in range(rounds):
                b_baselines = {b: baselines[backend_to_device[b]][c_i][r_i] for b in bs} 
                for s_policy in policies:
                    policy = POLICIES[s_policy]
                    s_weights = policy({b: backend_to_device[b] for b in bs}, calibration_report)
                    for b in bs:
                        if b not in s_weights:
                            s_weights[b] = 0
                    for m_policy in policies:
                        policy = POLICIES[m_policy]
                        m_weights = policy({b: backend_to_device[b] for b in bs}, calibration_report)
                        for b in bs:
                            if b not in m_weights:
                                m_weights[b] = 0
                        disptach = {b: math.floor(w*shots) for b,w in s_weights.items()}
                        _tot_shots = sum(disptach.values())
                        if _tot_shots < shots:
                            for _ in range(shots - _tot_shots):
                                disptach[random.choice(list(disptach.keys()))] += 1
                        
                        counts = {}
                        for b,s in disptach.items():
                            baseline = b_baselines[b]
                            counts[b] = sample_counts(baseline, s)
                            assert sum(counts[b].values()) == s
                            
                        merged = {}
                        for b,w in m_weights.items():
                            for k,v in counts[b].items():
                                if k not in merged:
                                    merged[k] = 0
                                merged[k] += v*w
                                
                        tot_weights = sum(m_weights.values())
                        for k in merged.keys():
                            merged[k] = merged[k]/tot_weights
                        
                        sum_values = sum(merged.values())
                        for k in merged.keys():
                            merged[k] = merged[k]/sum_values
                            
                        for k in merged.keys():
                            merged[k] *= shots
                            merged[k] = int(round(merged[k])) 
                            
                        experiments.append({
                            "backends": {b: backend_to_device[b] for b in bs},
                            "s_policy": s_policy,
                            "s_weights": s_weights,
                            "m_policy": m_policy,
                            "m_weights": m_weights,
                            "counts": counts,
                            "merged": merged,
                            "circuit": circuits[c_i],
                            "dispatch": disptach,
                            "round": r_i,
                        })
    
    return experiments

def run_experiments(now, baselines, calibration_report, backend_to_device):
    debug("Running experiments...")
    experiments_now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    os.makedirs(f"experiments/{now}/experiments/{experiments_now}", exist_ok=True)
    os.makedirs(f"experiments/{now}/experiments/{experiments_now}/docs", exist_ok=True)
    os.system(f"cp config.ini run.py experiments/{now}/experiments/{experiments_now}/docs")
    
    json.dump(calibration_report, open(f"experiments/{now}/experiments/{experiments_now}/calibration_report.json", "w"), indent=4, default=np_encoder)
    
    policies = json.loads(CONFIG["EXPERIMENT"]["policies"])
    
    config = configparser.ConfigParser()
    config.optionxform=str # Preserve case
    config.read(f"experiments/{now}/docs/config.ini")
    circuits = json.loads(config["EXPERIMENT"]["circuits"])
    
    backends = list(backend_to_device.keys())
    
    experiments = perform_experiments(powerset(backends), baselines, calibration_report, backend_to_device, policies, circuits)
                
    json.dump(experiments, open(f"experiments/{now}/experiments/{experiments_now}/experiments.json", "w"), indent=4, default=np_encoder)
    
    return experiments
            
def analyse_experiment(baseline_dir, experiment_dir):
    import pandas as pd
    debug("Analysing experiment...")
    analysis_now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    os.makedirs(f"experiments/{baseline_dir}/experiments/{experiment_dir}/analysis/{analysis_now}", exist_ok=True)
    os.makedirs(f"experiments/{baseline_dir}/experiments/{experiment_dir}/analysis/{analysis_now}/docs", exist_ok=True)
    os.system(f"cp config.ini run.py experiments/{baseline_dir}/experiments/{experiment_dir}/analysis/{analysis_now}/docs")
    
    baselines = json.load(open(f"experiments/{baseline_dir}/baselines/baselines.json"))
    calibration_report = json.load(open(f"experiments/{baseline_dir}/experiments/{experiment_dir}/calibration_report.json"))
    experiments = json.load(open(f"experiments/{baseline_dir}/experiments/{experiment_dir}/experiments.json"))
    
    config = configparser.ConfigParser()
    config.optionxform=str # Preserve case
    config.read(f"experiments/{baseline_dir}/docs/config.ini")
    circuits_names = json.loads(config["EXPERIMENT"]["circuits"])
    
    experiment_circuits = []
    for c_f in circuits_names:
        with open(f"experiments/{baseline_dir}/circuits/{c_f}.qasm", "r") as f:
            experiment_circuits.append(QuantumCircuit.from_qasm_str(f.read()))
            
    ground_truth = compute_ground_truth(experiment_circuits)
    json.dump(ground_truth, open(f"experiments/{baseline_dir}/experiments/{experiment_dir}/analysis/{analysis_now}/ground_truth.json", "w"))
    
    data = []
    for backend, _baselines in baselines.items():
        for c_i,c_data in enumerate(_baselines):
            for r_i, c in enumerate(c_data):
                dist, dist_err = hellinger_distance(c, ground_truth[c_i], len(list(c.keys())[0]))
                data.append({
                    "backends_len": 1,
                    "backends": backend,
                    "circuit": circuits_names[c_i],
                    "distance": dist,
                    "distance_err": dist_err,
                })
                
    for experiment in experiments:
        dist, dist_err = hellinger_distance(experiment["merged"], ground_truth[circuits_names.index(experiment["circuit"])], len(list(experiment["merged"].keys())[0]))
        data.append({
            "backends_len": len(experiment["backends"]),
            "backends": "/".join([b for b in experiment["backends"].values()]),
            "circuit": experiment["circuit"],
            "s_policy": experiment["s_policy"],
            "m_policy": experiment["m_policy"],
            "distance": dist,
            "distance_err": dist_err,
        })
        
    df = pd.DataFrame(data)
    df.to_csv(f"experiments/{baseline_dir}/experiments/{experiment_dir}/analysis/{analysis_now}/analysis.csv", index=False)
    
    config = configparser.ConfigParser()
    config.optionxform=str # Preserve case
    config.read(f"experiments/{baseline_dir}/experiments/{experiment_dir}/docs/config.ini")
    
    policies = json.loads(config["EXPERIMENT"]["policies"])
    
    backends = list(baselines.keys())
    backend_to_device = {}
    bss = []
    for b in backends:
        bs = [b+"_0"]
        backend_to_device[b+"_0"] = b
        i = 1
        for _ in range(len(backends)-1):
            bs.append(b+"_"+str(i))
            backend_to_device[b+"_"+str(i)] = b
            i += 1
            bss.append(bs.copy())
    
    control_group_data = perform_experiments(bss, baselines, calibration_report, backend_to_device, policies, circuits_names)
    
    json.dump(control_group_data, open(f"experiments/{baseline_dir}/experiments/{experiment_dir}/analysis/{analysis_now}/control_group.json", "w"), indent=4, default=np_encoder)
    
    cg_data = []
    for experiment in control_group_data:
        dist, dist_err = hellinger_distance(experiment["merged"], ground_truth[circuits_names.index(experiment["circuit"])], len(list(experiment["merged"].keys())[0]))
        cg_data.append({
            "backends_len": len(experiment["backends"]),
            "backends": "/".join([b for b in experiment["backends"].values()]),
            "circuit": experiment["circuit"],
            "s_policy": experiment["s_policy"],
            "m_policy": experiment["m_policy"],
            "distance": dist,
            "distance_err": dist_err,
        })
        
    cg_df = pd.DataFrame(cg_data)
    cg_df.to_csv(f"experiments/{baseline_dir}/experiments/{experiment_dir}/analysis/{analysis_now}/control_group_analysis.csv", index=False)
    
    return df, cg_df
            
def init():
    global tr, vp, disp, cm, provider
    global noise_models, properties, backends, initial_placement
    
    debug("Initialising...")
    
    tr = Translator()
    vp = VirtualProvider(TOKENS)
    disp = Dispatcher(vp, tr)
    cm = CompilationManager()
    provider=IBMProvider()

    noise_models = get_noise_models()
    properties = get_properties()
    backends = get_backends(blueprint, noise_models)
    
    debug("Backends ready!")
    
    _initial_placement = {}
    for backend, placements in INITIAL_PLACEMENT.items():
        _initial_placement[backend] = {}
        for size, placement in placements.items():
            if size == qubits:
                _initial_placement[backend] = placement
                break
            elif size > qubits:
                _initial_placement[backend] = placement[:qubits]
                break
    initial_placement = _initial_placement
    
                
if __name__ == "__main__":
    qubits = int(CONFIG["GLOBAL"]["qubits"])
    debug("*** QUBITS:", qubits)
    
    blueprint = {k:int(v) for k,v in CONFIG["BLUEPRINT"].items() if int(v) > 0}
    backend_to_device = {}
    for b,i in blueprint.items():
        if not b.startswith("ibm"):
            b = "simulator_"+b
        backend_to_device[b+"_"+str(i)] = b
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if len(sys.argv) > 2:
            phase = sys.argv[2]
        else:
            phase = None
    else:
        mode = None
        phase = None
            
    debug("*** MODE:", mode)
    debug("*** PHASE:", phase)
    try:
        debug("*** OTHER ARGS:", sys.argv[3:])
    except:
        debug("*** OTHER ARGS:", "[]")
    
    
    new_calibration = False
    now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    
    if mode is not None:
        if mode.lower().startswith("cal") or mode.lower() == "c":
            circuits = setup_calibration(now, existing_circuits=True)   
            if phase is not None:
                if phase.lower().startswith("count") or phase.lower().startswith("c"):
                    init()
                    counts = get_calibration_counts(now, [c.copy() for c in circuits], backends, initial_placement)
                elif phase.lower().startswith("an") or phase.lower().startswith("a"):
                    os.rmdir(f"calibrations/data/{now}")
                    if len(sys.argv) > 3:
                        counts = json.load(f"calibrations/data/{sys.argv[3]}/counts/counts.json")
                    else:
                        dirs = os.listdir("calibrations/data")
                        dirs = [f for f in dirs if f != "circuits" and os.path.isdir(f"calibrations/data/{f}")]
                        counts_dirs = sorted(dirs, key=lambda x: datetime.datetime.strptime(x, "%Y-%m-%d--%H-%M-%S"), reverse=True)
                        debug("Using counts from:", counts_dirs[0])
                        counts = json.load(open(f"calibrations/data/{counts_dirs[0]}/counts/counts.json"))
                    
                    calibration_report = analyse_calibration_counts(counts_dirs[0], counts, circuits)    
                else:
                    debug("Invalid phase!")
                    exit(1)
            else:
                init()
                counts = get_calibration_counts(now, [c.copy() for c in circuits], backends, initial_placement)
                calibration_report = analyse_calibration_counts(now, counts, circuits)
                
        elif mode.lower().startswith("exp") or mode.lower() == "e":
            if phase is not None:
                if phase.lower().startswith("bas") or phase.lower().startswith("b"):
                    init()
                    circuits = setup_experiment(now)
                    baselines = get_baselines(now, circuits, qubits)
                    
                elif phase.lower().startswith("count") or phase.lower().startswith("c"):
                    if len(sys.argv) > 3:
                        baselines = json.load(open(f"experiments/{sys.argv[3]}/baselines/baselines.json"))
                        now = sys.argv[3]
                        already_init = False
                    else:   
                        init()
                        already_init = True
                        circuits = setup_experiment(now)
                        baselines = get_baselines(now, circuits, qubits)
                        
                    if len(sys.argv) > 4:
                        calibration_folder = sys.argv[4]
                        most_recent_analysis = sorted(os.listdir(f"calibrations/data/{calibration_folder}/analysis"), key=lambda x: datetime.datetime.strptime(x, "%Y-%m-%d--%H-%M-%S"), reverse=True)[0]
                        calibration_report = json.load(open(f"calibrations/data/{calibration_folder}/analysis/{most_recent_analysis}/calibration_report.json"))
                    else:
                        if not already_init:
                            init()
                        _circuits = setup_calibration(now)
                        counts = get_calibration_counts(now, [c.copy() for c in _circuits], backends, initial_placement)
                        calibration_report = analyse_calibration_counts(now, counts, _circuits)
                        new_calibration = True
                        
                    experiments_data = run_experiments(now, baselines, calibration_report, backend_to_device)
                    experiments_now = sorted(os.listdir(f"experiments/{now}/experiments"), key=lambda x: datetime.datetime.strptime(x, "%Y-%m-%d--%H-%M-%S"), reverse=True)[0]
                    analyse_experiment(now, experiments_now)
                        
                elif phase.lower().startswith("an") or phase.lower().startswith("a"):
                    if len(sys.argv) > 3:
                        baseline_dir = sys.argv[3]
                    else:
                        baseline_dir = sorted(os.listdir("experiments"), key=lambda x: datetime.datetime.strptime(x, "%Y-%m-%d--%H-%M-%S"), reverse=True)[0]
                        
                    if len(sys.argv) > 4:
                        experiment_dir = sys.argv[4]
                    else:
                        experiment_dir = sorted(os.listdir(f"experiments/{baseline_dir}/experiments"), key=lambda x: datetime.datetime.strptime(x, "%Y-%m-%d--%H-%M-%S"), reverse=True)[0]
                        
                    analyse_experiment(baseline_dir, experiment_dir)
                else:
                    debug("Invalid phase!")
                    exit(1)
            else:
                init()
                if len(sys.argv) > 3:
                    calibration_folder = sys.argv[3]
                    most_recent_analysis = sorted(os.listdir(f"calibrations/data/{calibration_folder}/analysis"), key=lambda x: datetime.datetime.strptime(x, "%Y-%m-%d--%H-%M-%S"), reverse=True)[0]
                    calibration_report = json.load(open(f"calibrations/data/{calibration_folder}/analysis/{most_recent_analysis}/calibration_report.json"))
                else:
                    _circuits = setup_calibration(now)
                    counts = get_calibration_counts(now, [c.copy() for c in _circuits], backends, initial_placement)
                    calibration_report = analyse_calibration_counts(now, counts, _circuits)
                    new_calibration = True
                
                circuits = setup_experiment(now)
                baselines = get_baselines(now, circuits, qubits)
                experiments_data = run_experiments(now, baselines, calibration_report, backend_to_device)
                experiments_now = sorted(os.listdir(f"experiments/{now}/experiments"), key=lambda x: datetime.datetime.strptime(x, "%Y-%m-%d--%H-%M-%S"), reverse=True)[0]
                analyse_experiment(now, experiments_now)
        else:
            debug("Invalid mode!")
            exit(1)
    else:
        init()
        circuits = setup_calibration(now)
        counts = get_calibration_counts(now, [c.copy() for c in circuits], backends, initial_placement)
        calibration_report = analyse_calibration_counts(now, counts, circuits)
        circuits = setup_experiment(now)
        baselines = get_baselines(now, circuits, qubits)
        experiments_data = run_experiments(now, baselines, calibration_report, backend_to_device)
        experiments_now = sorted(os.listdir(f"experiments/{now}/experiments"), key=lambda x: datetime.datetime.strptime(x, "%Y-%m-%d--%H-%M-%S"), reverse=True)[0]
        analyse_experiment(now, experiments_now)
        new_calibration = True
        
    if len(circuits_to_remove) > 0:
        for c in circuits_to_remove:
            os.remove(c)
            
    if new_calibration:
        shutil.rmtree(f"calibrations/data/{now}")
            