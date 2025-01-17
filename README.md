# MF-GLFGCN:Multi-stream global-local fusion graph convolutional network for skeleton-based gait recognition
*We open source all the models, methods and datasets mentioned in the paper, and hope to provide some reference value for gait recognition research.*

## Visualization of collected data:
![image](https://github.com/Xu-repository/MF-GLFGCN/blob/master/img/data%20visualization.gif)

## The data collection scene:
![image](https://github.com/Xu-repository/MF-GLFGCN/blob/master/img/scene.png)

### Skeleton dataset path: 
```shell
cd ./data/new_data_plus_2/train_data/*.csv
```
### Skeleton dataset analysis:
Taking part of the ../train_data/0.csv file as an example.
|        |    0 |   1 |   2 |   4 |   5 |   6 |
|:-------|-----:|-----:|-----:|-----:|-----:|-----:|
| 0 | -74.02 | -310.438 | 1833.24   | -80.907 | -492.666 | 1821.558 |
| 1 | -64.242 | -295.929 | 1822.069 | -70.83 | -478.592 | 1810.891 |
| 2 | -51.235 | -294.411 | 1831.823 | -60.415 | -479.205 | 1817.587  |
| 3 | -43.467 | -293.231 | 1830.682 | -55.04 | -478.889 | 1817.587  |
| 4 | -32.085 | -290.424 | 1830.682 | -41.851 | -476.189 | 1814.108  |
| 5 | -19.257 | -292.719 | 1823.502 | -31.681 | -477.564 | 1806.558  |
| 6 | -18.434 | -296.251 | 1819.715 | -28.079 | -481.906 |  1804.226 |

Each row represents a frame of skeleton data, Each of the three columns represents a joint with 3D coordinates(x, y and z axis) in millimeters.
See the link for a detailed skeleton data introduction.[body-joints](https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/body-joints )

## The framework of the Multi-Stream Global-Local Fusion Graph Convolutional Network (MS-GLFGCN):
![image](https://github.com/Xu-repository/MF-GLFGCN/blob/master/img/model.png)

## Reproduction process
### Prerequisites
- Python >= 3.6
- CUDA >= 10

First, create a virtual environment or install dependencies directly with:
```shell
pip3 install -r requirements.txt
```
### Data preparation
The preprocessed skeleton sequence is placed in ./data/new_data_plus_2

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

### Main Results
**Model1：*MS-GLFGCN***:&nbsp;The results of the moedl on the skeleton dataset.

Parameter :&nbsp;0.51M &emsp;	FLOPs:&nbsp;7.52G&emsp;	Mean Acc.:&nbsp;98.53%&emsp;	Max Acc.:&nbsp;98.53%&emsp;\
\
**Model2：*GaitSet_MS***:&nbsp;The average accuracy of GaitSet with our method on the skeleton-based CASIA-B.
|        |    0 |   18 |   36 |   54 |   72 |   90 |   108 |   126 |   144 |   162 |   180 |   mean |
|:-------|-----:|-----:|-----:|-----:|-----:|-----:|------:|------:|------:|------:|------:|-------:|
| NM#5-6 | 74.2 | 81.1 | 83.2   | 82.8 | 79.7 | 82 |  83.8 |  87 |  86.6 |  86 |  73.9 |   81.85 |
| BG#1-2 | 63.5 | 69.2 | 68.5 | 68.1 | 65.5 | 64.3 |  66.6   |  68.1 |  65.4 |  68.1 |  55.7 |   65.73 |
| CL#1-2 | 50.9 | 56.7 | 55.2 | 51.9 | 53.3 | 52   |  52.1 |  49.6 |  49.5 |  50.6 |  44.1 |   51.45 |

The average accuracy of GaitSet with our method on the skeleton-based OU-MVLP.
|        |    0 |   15 |   30 |   45 |   60 |   75 |   90 |   180 |   195 |   210 |   225 | 240| 255| 270|   mean |
|:-------|-----:|-----:|-----:|-----:|-----:|-----:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|
| Acc. | 41.99 | 56.23 | 61.88   | 64.39 | 62.43 | 59.17 |  52.76 |  36.75 | 47.63 | 47.02 |  59.63 |   58.44 | 54.9| 48.84|53.72 |

## Licence & Acknowledgement

The following parts of the code are borrowed from other projects. Thanks for their wonderful work!
- Object Detector: [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
- Pose Estimator: [HRNet/HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation)
- ST-GCN Model: [yysijie/st-gcn](https://github.com/yysijie/st-gcn)
- ResGCNv1 Model: [yfsong0709/ResGCNv1](https://github.com/yfsong0709/ResGCNv1)
- SupCon Loss: [HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast)
- GaitGraph: [tteepe/GaitGraph](https://github.com/tteepe/GaitGraph)
- OpenGait: [ShiqiYu/OpenGait](https://github.com/ShiqiYu/OpenGait)

*Note: This code is only used for academic purposes, people cannot use this code for anything that might be considered commercial use.*
