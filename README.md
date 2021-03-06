# Pytorch Fully Convolutional Network

Paper implementation of [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

## Requirements

- Python 3.6
- PyTorch

## Installation

```
git clone https://github.com/andersskog/pytorch-fully-convolutional-network.git
cd pytorch-fully-convolutional-network
pip3 install -r requirements.txt
```
Note: for MacOS, see http://pytorch.org/ for other OS.

## Dataset

Images and annotations are a [subset of ADE20K Dataset](http://sceneparsing.csail.mit.edu/). This dataset is build to predict a total of 150 classes.

## Usage

```
python3 main.py train_list val_list dataset_path
```

### Example

```
python3 main.py train.txt val.txt data
```