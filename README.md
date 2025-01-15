# MF-GLFGCN:Multi-stream global-local fusion graph convolutional network for skeleton-based gait recognition
The data collection scene:
![image](https://github.com/Xu-repository/MF-GLFGCN/blob/master/img/scene.png)

Visualization of collected data:
![img](https://github.com/Xu-repository/MF-GLFGCN/blob/master/img/video1.gif)

The framework of the Multi-Stream Global-Local Fusion Graph Convolutional Network (MS-GLFGCN):
![image](https://github.com/Xu-repository/MF-GLFGCN/blob/master/img/model.png)

This repository contains the PyTorch code for:

### Prerequisites
- Python >= 3.6
- CUDA >= 10

First, create a virtual environment or install dependencies directly with:
```shell
pip3 install -r requirements.txt
```
### Data preparation
The preprocessed skeleton sequence is placed in ./data

### Train
To train the model you can run the `train.py` script. To see all options run:
```shell
cd src
export PYTHONPATH=${PWD}:$PYTHONPATH

python3 train.py ./data/new_data_plus_2/train_data \
                 --./data/new_data_plus_2/valid_data \
                 --batch_size 16 \
                 --batch_size_validation 16 \
                 --embedding_layer_size 68 \
                 --epochs 100\
                 --learning_rate 1e-5 \
                 --temp 0.01 \
	               --use_multi_branch \
                 --network_name resgcn-n39-r8
```

Check `experiments/1_train_*.sh` to see the configurations used in the paper. 

Optionally start the tensorboard with: 
```shell
tensorboard --logdir=save/casia-b_tensorboard 
```

### Evaluation
Evaluate the models using `evaluate.py` script. To see all options run:
```shell
python3 evaluate.py --help
```

## Main Results
Top-1 Accuracy per probe angle excluding identical-view cases for the provided models on dataset.

Parameter :0.51M	FLOPs:7.52G	Mean Acc.:98.53%	Max Acc.:98.53%

## Licence & Acknowledgement

The following parts of the code are borrowed from other projects. Thanks for their wonderful work!
- Object Detector: [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
- Pose Estimator: [HRNet/HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation)
- ST-GCN Model: [yysijie/st-gcn](https://github.com/yysijie/st-gcn)
- ResGCNv1 Model: [yfsong0709/ResGCNv1](https://github.com/yfsong0709/ResGCNv1)
- SupCon Loss: [HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast)
- GaitGraph: [tteepe/GaitGraph](https://github.com/tteepe/GaitGraph)
- OpenGait: [ShiqiYu/OpenGait](https://github.com/ShiqiYu/OpenGait)

