## Learning DAG-Based Job Scheduling with Graph Neural Networks and Reinforcement Learning

This project explores **job scheduling on Directed Acyclic Graphs (DAGs)** using a combination of **Graph Neural Networks (GNNs)** and **Reinforcement Learning (RL)**.

The main idea is to learn a scheduling policy that assigns tasks to machines efficiently while:
- Respecting DAG precedence constraints.
- Reducing makespan or other latency metrics.
- Improving overall throughput and resource utilization.

The core workflow lives in the Jupyter notebook `RL-GNN Scheduler.ipynb`, with preprocessed data in `data_clean/` and trained models in `models/`.

---

## Project Structure

- **`RL-GNN Scheduler.ipynb`**  
  Main notebook that typically:
  - Loads the cleaned DAG / job data from `data_clean/`.
  - Builds the GNN-based representation of the scheduling problem.
  - Defines and trains an RL agent that uses GNN embeddings.
  - Evaluates the learned scheduler against baseline heuristics (if implemented).

- **`data_clean/`**
  - **`batch_instance.csv`** – Job/batch-level information (e.g., job IDs, arrival times, priorities, etc.).
  - **`batch_task.csv`** – Task-level information, including DAG structure (e.g., task IDs, predecessors, durations, and possibly resource needs).
  - **`machine_meta.csv`** – Machine-level metadata (e.g., machine IDs, capacities, or other static features).

- **`data_clean.ipynb`**  
  Notebook used to clean and preprocess raw data into the CSVs in `data_clean/`, for example:
  - Parsing raw logs or original datasets.
  - Constructing DAGs from dependency information.
  - Generating task, job, and machine features.

- **`models/`**
  - **`gnn_only_model.pth`** – Trained PyTorch model for the GNN encoder / predictor (without full RL control loop).
  - **`rl_model.pth`** – Trained PyTorch model for the RL-based scheduling policy (possibly including GNN components).

---

## Key Concepts

- **DAG-Based Job Scheduling**  
  Jobs are modeled as DAGs where:
  - Nodes represent tasks.
  - Directed edges represent precedence constraints (a parent task must finish before its child starts).

- **Graph Neural Networks (GNNs)**  
  A GNN encodes the DAG and its node/edge features into embeddings that capture:
  - Task readiness and criticality.
  - Position in the DAG (e.g., near the critical path).
  - Global context about the current job and resource state.

- **Reinforcement Learning (RL)**  
  An RL agent interacts with a scheduling environment:
  - **State**: Encoded DAG, task states (ready/running/completed), machine loads, and other system context.
  - **Action**: Choose which task(s) to schedule next and/or on which machine(s).
  - **Reward**: Designed to encourage good scheduling behavior, e.g., minimizing makespan, tardiness, or idle time.

---

## Typical Workflow

1. **Data Preparation**
   - Run `data_clean.ipynb` (if you need to regenerate data).
   - This should produce or refresh:
     - `data_clean/batch_instance.csv`
     - `data_clean/batch_task.csv`
     - `data_clean/machine_meta.csv`

2. **Model Training & Evaluation**
   - Open `RL-GNN Scheduler.ipynb`.
   - Load data from `data_clean/`.
   - Build graphs and features for the GNN.
   - Define and train:
     - A GNN encoder (`gnn_only_model.pth`).
     - An RL policy (`rl_model.pth`) that uses these embeddings.
   - Evaluate performance and compare against simple heuristics (e.g., FIFO, SJF).

3. **Using Pretrained Models**
   - Load `models/gnn_only_model.pth` and `models/rl_model.pth` from the notebook.
   - Run pure evaluation without retraining to:
     - Quickly reproduce results.
     - Test new datasets or what-if scenarios.

---

## Configuration & Extensions

- **Changing the Objective**
  - Modify the reward function in the RL environment to:
    - Minimize makespan.
    - Penalize tardiness for high-priority jobs.
    - Encourage balanced machine utilization or fairness.

- **Adding Features**
  - Extend CSVs with additional:
    - Task features (e.g., memory, I/O intensity, priority).
    - Machine features (e.g., CPU cores, GPUs, power constraints).
  - Update the GNN input dimensions and any preprocessing code accordingly.

- **Scaling Up**
  - For larger-scale experiments:
    - Use PyTorch `DataLoader`s for batched graph processing.
    - Enable GPU acceleration in PyTorch.
    - Add checkpointing for long RL training runs.

---

## Requirements:

- **Python**: 3.9+  
- **Core Libraries**:
  - `torch`
  - `torch-geometric`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `jupyter`
- **Optional RL Libraries** :
  - `gymnasium`
  - `stable-baselines3`
---


## Contact

- **Author**: (Qi Wu)  
- **Email**: (q1wu@stanford.edu)

- **Author**: (Emery Yu)  
- **Email**: (emeryyu@stanford.edu)
