# SGR
Source code for IJCAI-ECAI 2022 paper: [Document-level Relation Extraction via Subgraph Reasoning](https://www.ijcai.org/proceedings/2022/0601.pdf).
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
## Citation
```
@inproceedings{ijcai2022-601,
  title     = {Document-level Relation Extraction via Subgraph Reasoning},
  author    = {Peng, Xingyu and Zhang, Chong and Xu, Ke},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {4331--4337},
  year      = {2022},
  month     = {7},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2022/601},
  url       = {https://doi.org/10.24963/ijcai.2022/601},
}
```