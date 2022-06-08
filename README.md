# SGR
Source code for IJCAI-ECAI 2022 paper: Document-level Relation Extraction via Subgraph Reasoning.
## Environments
- Python 3.8.8
- Cuda 10.1
## Requirements
- PyTorch 1.7.1
- DGL 0.6.1
- Transformers 4.11.3
- NetworkX 2.7.1
## Preparation
Run `python preprocess.py` or directly download the preprocessed data files from [here](https://drive.google.com/file/d/1D4_BSe0Yd8WHBwwhHWfruHnnh1cfwMAD/view?usp=sharing) and put them into the `data/prepro_data` directory.
## Train
```
python train.py
```
## Test
```
python test.py
```
