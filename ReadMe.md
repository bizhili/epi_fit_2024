# Topology Fitting — Command Line Usage

This repository provides a simulation framework for topology fitting and epidemic modeling on random graphs.  
The main entry point is `run.py`, which accepts a range of configuration parameters.

---

## Usage

Before use, first create folder for logs and results:

```bash
cd project_folder
mkdir logs
mkdir results/AA
mkdir results/AB
mkdir results/BA
mkdir results/BB
mkdir results/infer2018
```

```bash
python run.py [-h] 
              [--randomGraph RANDOMGRAPH] 
              [--seed SEED] 
              [--n N] 
              [--strains STRAINS] 
              [--epoches EPOCHES] 
              [--device DEVICE]
              [--weightModel WEIGHTMODEL] 
              [--intense INTENSE] 
              [--R0Mean R0MEAN] [--R0Std R0STD] 
              [--tauMean TAUMEAN] [--tauStd TAUSTD]
              [--modelLoad MODELLOAD] 
              [--dense DENSE] 
              [--identicalf IDENTICALF] 
              [--wsProbability WSPROBABILITY]
              [--evaluateEvery EVALUATEEVERY] 
              [--epi EPI]
```

---

## Options

### General
- `-h, --help`  
  Show help message and exit.  
- `--seed SEED` *(int, default=10)*  
  Random seed (used for generating topology and data).  
- `--device DEVICE` *(str, default="cuda:0")*  
  Compute device (`cuda:0` or `cpu`).  
- `--epoches EPOCHES` *(int, default=100000)*  
  Maximum number of training epochs.  
- `--evaluateEvery EVALUATEEVERY` *(int)*  
  Frequency (in epochs) to run evaluation.  

### Graph Topology
- `--randomGraph RANDOMGRAPH` *(str, default="RGG")*  
  Random graph model. Options: `RGG`, `ER`, `WS`, `BA`.  
- `--n N` *(int, default=50)*  
  Number of nodes in the graph.  
- `--dense DENSE` *(int, default=8)*  
  Average degree for `BA`, `WS`, `ER`, `RGG`. If negative, uses `log(n)` density.  
- `--wsProbability WSPROBABILITY` *(float, default=0.1)*  
  Rewiring probability for the Watts–Strogatz (WS) model.  

### Weights & Selection
- `--weightModel WEIGHTMODEL` *(str, default="degree")*  
  Adjacency weight model. Options: `degree`, `gravity`, `identical`.  
- `--identicalf IDENTICALF` *(float, default=0.01)*  
  Constant weight factor for `identical` model.  
- `--intense INTENSE` *(int, default=0)*  
  Selection intensity for node degree. Options: `0`, `1`, `2`, `-1` (linear probing).  

### Epidemic Parameters
- `--strains STRAINS` *(int, default=1)*  
  Number of epidemic strains (1–4).  
- `--R0Mean R0MEAN` *(float, default=8.3)*  
  Mean of basic reproduction number \(R_0\).  
- `--R0Std R0STD` *(float, default=0.5)*  
  Standard deviation of \(R_0\).  
- `--tauMean TAUMEAN` *(float, default=6.2)*  
  Mean of infection duration \(	au\).  
- `--tauStd TAUSTD` *(float, default=0.1)*  
  Standard deviation of \(	au\).  

### Model
- `--modelLoad MODELLOAD` *(str, default="AA")*  
  Model to load. Options: `AA`, `AB`, `BA`, `BB`, `infer2018`, `ATA`.  
- `--epi EPI`  
  Path to empirical epidemic dataset for evaluation.  

---

## Example

```bash
python run.py --randomGraph RGG --strains 4 --n 100 --epoches 100000 --device cuda:0
```

---
