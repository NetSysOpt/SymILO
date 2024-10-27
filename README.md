# SymILO: A Symmetry-Aware Learning Framework for Integer Linear Optimization

This repository contains the code for the NeurIPS 2024 paper: **[SymILO: A Symmetry-Aware Learning Framework for Integer Linear Optimization](https://arxiv.org/abs/2409.19678)**.

## Environment Setup
To run this code, you need the following dependencies:
- Python 3.9
- CPLEX 22.2.0
- PyTorch 2.0.1

## Training the Model

To train the model, you can use the following bash commands:

```bash
dataset=IP # indicate the dataset
python train.py --expName $dataset --dataset $dataset --opt mean --epoch 50
```

## Evaluation

To evaluate the model, you can use the following commands:

```bash
dataset=IP
python eval.py --expName $dataset --dataset $dataset --method fixTop
python eval.py --expName $dataset --dataset $dataset --method PS
python eval.py --expName $dataset --dataset $dataset --method node_selection
```

## Citation

If you find this code helpful in your research, please consider citing our paper:

```bibtex
@article{chen2024symilo,
  title={SymILO: A Symmetry-Aware Learning Framework for Integer Linear Optimization},
  author={Chen, Qian and Zhang, Tianjian and Yang, Linxin and Han, Qingyu and Wang, Akang and Sun, Ruoyu and Luo, Xiaodong and Chang, Tsung-Hui},
  journal={NeurIPS},
  year={2024}
}
```
