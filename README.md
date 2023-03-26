# UAV’s Status Is Worth Considering: A Fusion Representations Matching Method for Geo-Localization
Paper Link : https://doi.org/10.3390/s23020720 (Open Access) 

## Experiment Result

| Method        | Image_Size | **Drone** → Satellite | **Drone** → Satellite | Satellite → Drone | Satellite → Drone |
| ------------- | ---------- | --------------------- | --------------------- | ----------------- | ----------------- |
|               |            | Recall@1              | AP                    | Recall@1          | AP                |
| Baseline      | 384*384    | 62.99                 | 67.69                 | 75.75             | 62.09             |
| LCM           | 384*384    | 66.65                 | 70.82                 | 79.89             | 65.38             |
| LPN           | 384*384    | 78.02                 | 80.99                 | 86.16             | 76.56             |
| LDRVSD        | 384*384    | 81.02                 | 83.51                 | 89.87             | 79.80             |
| SGM           | 256*256    | 82.14                 | 84.72                 | 88.16             | 81.81             |
| PCL           | 512*512    | 83.27                 | 87.32                 | 91.78             | 82.18             |
| FSRA          | 384*384    | 85.50                 | 87.53                 | 89.73             | 84.94             |
| MSBA          | 384*384    | 86.61                 | 88.55                 | 92.15             | 84.54             |
| **MBF(ours)** | 384*384    | 89.05                 | 90.61                 | 93.15             | 88.17             |

## Quick Start
### Installation
Install Pytorch and Torchvision https://pytorch.org/get-started/locally/

install other libs
```shell
pip install timm pyyaml pytorch-metric-learning scipy pandas grad-cam pillow pytorch_pretrained_bert

```
### Generate word embeddings for University-1652
University-1652 Dataset Link https://github.com/layumi/University1652-Baseline
set correct dataset path in settings.yaml, then run
```shell
python U1652_bert.py
```

### Dataset files form
University-1652 dir tree:
```text

├── University-1652/
│   ├── readme.txt
│   ├── train/
│       ├── drone/                   /* drone-view training images 
│           ├── 0001
|           ├── 0002
|           ...
│       ├── street/                  /* street-view training images 
│       ├── satellite/               /* satellite-view training images       
│       ├── google/                  /* noisy street-view training images (collected from Google Image)
│       ├── text_drone/              /* word embeddings
|           ├── image-01.pth
|           ├── image-02.pth
|           ...
│       ├── text_satellite/ 
|           ├── satellite.pth
│   ├── test/
│       ├── query_drone/  
│       ├── gallery_drone/  
│       ├── query_street/  
│       ├── gallery_street/ 
│       ├── query_satellite/  
│       ├── gallery_satellite/ 
│       ├── 4K_drone/
│       ├── text_drone/              /* word embeddings
|           ├── image-01.pth
|           ├── image-02.pth
|           ...
│       ├── text_satellite/ 
|           ├── satellite.pth
```

### Train
```shell
python train.py --cfg "settings.yaml"
```
Config file (settings.yaml) sets parameter and path
```text
# dateset path
dataset_path: /home/sues/media/disk1/University-Release-MultiModel/University-Release
weight_save_path: /home/sues/save_model_weight

# apply LPN and set block number
LPN : 1
block : 2

# super parameters
batch_size : 16
num_epochs : 80
drop_rate : 0.35
weight_decay : 0.0001
lr : 0.01

#intial parameters
image_size: 384
fp16 : 1
classes : 701

model : MBF
name: MBF_1652_2022-11-15-18:56:39 
```


### Test and evaluate
```shell
python U1652_test_and_evaluate.py --cfg "settings.yaml" --name "your_weight_dirname_1652_2022-11-16-15:14:14" --seq 1
```



### Best Weights
Please check the Release page
Best weights have been uploaded

Any questions or suggestions feel free to contact me 
email : m025120503@sues.edu.cn

## Citation

```text
@Article{uav2023zhu,
AUTHOR = {Zhu, Runzhe and Yang, Mingze and Yin, Ling and Wu, Fei and Yang, Yuncheng},
TITLE = {UAV&rsquo;s Status Is Worth Considering: A Fusion Representations Matching Method for Geo-Localization},
JOURNAL = {Sensors},
VOLUME = {23},
YEAR = {2023},
NUMBER = {2},
ARTICLE-NUMBER = {720},
URL = {https://www.mdpi.com/1424-8220/23/2/720},
PubMedID = {36679517},
ISSN = {1424-8220},
DOI = {10.3390/s23020720}
```
}
