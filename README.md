# E2EMap
Artifact Evaluation Reproduction for "E2EMap: End-to-End Reinforcement Learning for CGRA Compilation via Reverse Mapping",HPCA 2024.

## Table of contents
1. [Directory Structure](#directory-structure)
2. [Getting Started](#getting-started)
    1. [Hardware pre-requisities](#hardware pre-requisities)
    2. [Software pre-requisites](#software pre-requisites)
    3. [Installation][#installation]
    4. [Running example](#running-example)
    5. [Modify the parameters](#modify-the-parameters)
    6. [Data formats](#data-formats)

# Directory Structure

```
E2EMap
│   README.md
│   Agent.py
│   config.py (Read the configuration from the script)
│   dataGenerator.py (Generate dataset)
│   environment_routing.py (Including state, action, reward, etc)
│   graph_embedding.py (DFG and CGRA embedding)
│   main.py
│   utils.py (Do some preprocessing work)
│   Networks.py (Network structure)
│   run.sh (Run script)
│───data (Graph data)
│───saving_log (Save the Results)
│     │───saving_log_0grf_4lrf (Save the Results(0-register GRF and 4-register LRF))
│     │───saving_log_2grf_2lrf (Save the Results(2-register GRF and 2-register LRF))
│     │───saving_log_4grf_0lrf (Save the Results(4-register GRF and 0-register LRF))
│     │───saving_log_re_1 (Save the results of the mesh)
│     │───saving_log_re_2 (Save the results of the torus)
│     │───saving_log_re_3 (Save the results of the diagonal)
│     │───saving_log_re_4 (Save the results of the 1-hop)
│     │───saving_log_re_6 (Save the results of the Morphosys)
│     │───saving_log8_8 (Save the results of the 1-hop(8X8))
│     └───saving_log12_12 (Save the results of the 1-hop(12X12))
└───script
      │   reward_mode1.sh (script for mesh)
      │   reward_mode2.sh (script for torus)
      │   reward_mode3.sh (script for diagonal)
      │   reward_mode4.sh (script for 1-hop)
      │   reward_mode6.sh (script for Morphosys)
      │   run_all_0grf_4lrf.sh (script for mesh(0-register GRF and 4-register LRF))
      │   run_all_2grf_2lrf.sh (script for mesh(2-register GRF and 2-register LRF))
      │   run_all_4grf_0lrf.sh (script for mesh(4-register GRF and 0-register LRF))
      │   run_all_8_8.sh (script for 1-hop(8X8))
      │   run_all_12_12.sh (script for 1-hop(12X12))
```

# Getting started
## Hardware pre-requisities
* Ubuntu (we have tested Ubuntu 18.04)
* GPU (we use GeForce RTX 2080 Ti)

## Software pre-requisites
* python3.7
* tensorflow2.6.0

## Installation
We use anaconda to deploy the environment
```
conda create -n tf python=3.7
conda activate tf
pip install tensorflow==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
```
If there were no packages such as keras, scipy, and pygmtools, you can
```
pip install keras==2.6
pip install scipy
pip install pygmtools
```

## Running Example
You can use the following command to test
```
bash run.sh demo
```
The scripts in the script folder are executed continuously, you can
```
cd script
bash reward_mode4.sh
```
The adi and h2v2 operators have been taken out separately, as their experimental time is relatively long. You can test them through script run_adi_and_h2v2.sh.

## Modify the parameters
If you want to modify the parameters of the code, open the script file and modify the specified parameter information
The following is an explanation of some key parameters
* src_file_path(For example, data/adi.txt is the adi kernel)
* actor_lr(Representation learning rate)
* gcn_dims(Indicates the hidden layer dimension)
* max_iteration(Indicates the maximum number of iterations)
* batch_size(Represents the number of samples in a batch)
* pea_width(Indicates the scale of the CGRA. For example, 4 indicates that the scale of the CGRA is 4x4)
* reward_mode(Indicates the CGRA structure, 1 indicates the mesh structure, 2 indicates the torus structure, 3 indicates the Diagonal structure, and 4 indicates the 1-Hop structure, and 5 indicates the 1-Hop+Diagonal+torus structure, 6 indicates the 1-Hop+torus structure)
* max_LRF(the number of LRF resources in one FU)
* max_GRF(the number of GRF resources in a time slot)

## Data Formats
Each line of the input file indicates a node in a DFG, which includes 13 segments defined as follows:
```
|----------|------------|-------------|------------|-------------|------------|-------------|------------|-------------|---------------------|-------------------|-------|---------------|
|node index|child node 1|edge 1's type|child node 2|edge 2's type|child node 3|edge 3's type|child node 4|edge 4's type|earliest control step|latest control step|special|zero in-degree?|
|----------|------------|-------------|------------|-------------|------------|-------------|------------|-------------|---------------------|-------------------|-------|---------------|
```

For example : \
<img src="DFG.png" alt="drawing" width="100"/> \
the input data file should be:
```
1,2,0,4,0,0,0,0,0,0,0,0,0
2,3,0,0,0,0,0,0,0,1,1,0,1
3,4,0,5,0,0,0,0,0,2,2,0,1
4,5,0,0,0,0,0,0,0,3,3,0,1
5,0,0,0,0,0,0,0,0,4,4,0,1
6,3,0,0,0,0,0,0,0,0,1,0,0
```
