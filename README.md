# About
This repository builds on the the official PyTorch implementation
of `Learning Intra-Batch Connections for Deep Metric Learning`.



We also support mixed-precision training via Nvidia Apex and describe how to use it in usage.

We support training on 4 datasets: CUB-200-2011, CARS 196, and In-Shop datasets.


# Set up


1. Clone and enter this repository:

        git clone https://github.com/Steven177/intra_batch

        cd intra_batch

2. Create an Anaconda environment for this project:
To set up a conda environment containing all used packages, please fist install 
anaconda and then run
    1.       conda env create -f environment.yml
    2.      conda activate intra_batch_dml

3. Download datasets:
Make a data directory by typing 

        mkdir data
    Then download the datasets using the following links and unzip them in the data directory:
    * CUB-200-2011: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
    * Cars196: https://vision.in.tum.de/webshare/u/seidensc/intra_batch_connections/CARS.zip
    * In-Shop: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html

    We also provide a parser for Stanford Online Products and In-Shop datastes. You can find dem in the `dataset/` directory. The datasets are expected to be structured as 
    `dataset/images/class/`, where dataset is either CUB-200-2011, CARS, Stanford_Online_Products or In_shop and class are the classes of a given dataset. Example for CUB-200-2011: 

            CUB_200_2011/images/001
            CUB_200_2011/images/002
            CUB_200_2011/images/003
            ...
            CUB_200_2011/images/200


4. Download our models: Please download the pretrained weights by using

        wget https://vision.in.tum.de/webshare/u/seidensc/intra_batch_connections/best_weights.zip
        wget https://vision.in.tum.de/webshare/u/seidensc/weights_for_StPh/cub_weights.zip
        wget https://vision.in.tum.de/webshare/u/seidensc/weights_for_StPh/cars_weights2.zip

    and unzip them.

# Usage
You can find config files for training and testing on each of the datasets in the `config/` directory. For training and testing, you will have to input which one you want to use (see below). You will only be able to adapt some basic variables over the command line. For all others please refer to the yaml file directly.
The config folder has several subfolders, which refer to different training and testing setups.

## Testing
To test to networks choose one of the config files for testing, e.g., `config_cars_test.yaml` to evaluate the performance on Cars196 and run:

    python train.py --config_path config_cars_test.yaml --dataset_path <path to dataset> 

The default dataset path is data.

## Training
To train a network choose one of the config files for training like `config_cars_train.yaml` to train on Cars196 and run:

    python train.py --config_path config_cars_train.yaml --dataset_path <path to dataset> --net_type <net type you want to use>

Again, if you don't specify anything, the default setting will be used. For the net type you have the following options:

`resnet18, resnet32, resnet50, resnet101, resnet152, densenet121, densenet161, densenet16, densenet201, bn_inception`

If you want to use apex add `--is_apex 1` to the command.


# Results 
## MPN during training  (R@1)
|             | CUB-200-2011 |   Cars196   | In-Shop|
| ------------- |:------|------:| ----- |
| Baseline  | 69.2  | 87.0 | 92.4  |
| w/o Attention       | 68.9  | 86.8  | 92.7 |
| w/o Feed-Forward       | 69.4  | 85.9  | 92.7 | 
| w/o Add & Norm       | 68.1  | 85.2  |  90.5 |
| Prenorm       | 68.1  | 87.0 | 92.2 |
| MLP Neck (instead 0f MPN)       | 68.6  | 82.7  | -| 


## MPN during testing  (R@1)
| Training Sampling | Testing Sampling | CUB-200-2011 | Cars196 |
| ------------- |-|-|:------|
| Labels | Backbone only | 69.2  | 87.0  |
| Labels |Labels | 72.0  | 85.7  |
| Labels | K-means | 69.3  | 76.2  |
| Labels | Reciprocal-KNN | 68.9  | 87.3  |
| Random | Random | 64.1  | 86.1  |
| K-means | K-means | 70.1  | 79.1  |
| Labels | DB-SCAN w JS | 51.4  | 42.4  |

# Citation

If you find this code useful, please consider citing the following paper:

```
@inproceedings{DBLP:conf/icml/SeidenschwarzEL21,
  author    = {Jenny Seidenschwarz and
               Ismail Elezi and
               Laura Leal{-}Taix{\'{e}}},
  title     = {Learning Intra-Batch Connections for Deep Metric Learning},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning,
               {ICML} 2021, 18-24 July 2021, Virtual Event},
  series    = {Proceedings of Machine Learning Research},
  volume    = {139},
  pages     = {9410--9421},
  publisher = {{PMLR}},
  year      = {2021},
}
```
