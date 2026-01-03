# Distributing Quantum Computations, Shot-wise

This repository contains the code used to distribute quantum computation workloads across multiple providers, along with calibration and analysis tooling for experiments.

## Reproducing the environment
- **Python:** 3.10 or later is recommended.
- **OS:** Linux or macOS with Bash and `python3` available.
- **Virtual environment (recommended):**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **Install packages:**
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

## Explicit reproduction packages
The following Python packages (and pinned versions) are required to reproduce the experiments. They are also listed in `requirements.txt` for installation via `pip install -r requirements.txt`.

- `cirq==1.3.0`
- `click==8.1.7`
- `mypy==1.8.0`
- `PennyLane==0.34.0`
- `pyquil==3.5.4`
- `pytket==1.24.0`
- `pytket-braket==0.34.1`
- `pytket-cirq==0.34.0`
- `pytket-pennylane==0.15.0`
- `pytket-pyquil==0.33.0`
- `pytket-qir==0.8.0`
- `pytket-qiskit==0.47.0`
- `qiskit==0.45.2`
- `requests==2.31.0`
- `mqt.bench==1.0.8`

## Configuration
Runtime settings live in `config.ini`. Set your access tokens under `[TOKENS]` (e.g., `IonQ`, `IBMQ`) and adjust experiment parameters such as `qubits`, calibration rounds, and policy selections under the corresponding sections.

## Running and reproducing experiments
1. (Optional) If you have compressed experiment data checked in, decompress them before running analysis:
   ```bash
   ./decompress.sh
   ```
2. Run a single experiment with logging via the helper script:
   ```bash
   ./experiment.sh exp
   ```
   The script delegates to `python3 -u run.py exp ...` and writes logs to `experiment.log`.
3. To iterate over all configurations in `configs/`, use:
   ```bash
   ./exps.sh
   ```
4. Compress generated `.csv` and `.json` artifacts before committing or archiving:
   ```bash
   ./compress.sh
   ```

`run.py` also supports calibration (`cal`) and analysis phases directly; refer to the inline documentation in the script for additional modes and arguments.

## License
This project is licensed under the Apache License, Version 2.0. See the `LICENSE` file for the full text.
