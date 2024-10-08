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

install other libs （timm should be 0.6.7, not latest）
```shell
pip install timm==0.6.7 pyyaml pytorch-metric-learning scipy pandas grad-cam pillow pytorch_pretrained_bert
```

### Generate word embeddings for University-1652
University-1652 Dataset Link https://github.com/layumi/University1652-Baseline


set correct dataset path in settings.yaml, then run
```shell
python U1652_bert.py
```

### Generate word embeddings for SUES-200
SUES-200 Dataset Link https://github.com/Reza-Zhu/SUES-200-Benchmark
Download SUES-200 Dataset and split dataset, set correct dataset path in settings.yaml, then run
```shell
python SUES_bert.py
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
SUES-200 dir tree:
```text
├── SUES-200/
│ ├── Training/
│     ├── 150
│          ├── drone/  /* drone-view training images 
│              ├── 0001  /* drone-view image of the first site: 50 images
│                  ├── 0.jpg
│                  ├── 1.jpg
│ 	          ...
│                  ├── 49.jpg
│              ├── 0002  /* drone-view image of the second site: 50 images
│                   ...
│          ├── satellite/  /* satellite-view training images 
│              ├── 0001  /* satellite-view image of the first site: 1 image
│                  ├── 0.png
│               ├── 0002  /* satellite-view image of the second site: 1 image
│                   ...
│          ├── text_drone 
│              ├── drone.pth /* word embeddings
│          ├── text_satellite
│              ├── satellite.pth /* word embeddings
│     ├── 200
│     ├── 250
│     ├── 300
│ ├── Testing/
│     ├── 150 
│             ├── query_drone/  /* drone-view query images 
│                 ├── 0008
│                        ...
│             ├── gallery_drone/ /* drone-view gallery images 
│                  ├── 0001
│                     ...
│                  ├── 0200
│             ├── query_satellite/    /* satellite-view query images
│             ├── gallery_satellite/  /* satellite-view gallery images
│             ├── text_drone
│                  ├── drone.pth
│             ├── text_satellite
│                  ├── satellite.pth
│     ├── 200
│     ├── 250
│     ├── 300  

```

### Train for University-1652
```shell
python train.py --cfg "settings.yaml"
```
Config file (settings.yaml) sets parameter and path
```yaml
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
### Train for SUES-200
```shell
python train.py --cfg "settings.yaml"
```
Config file (settings.yaml) sets parameter and path
```yaml

# dateset path
dataset_path: /home/LVM_date/zhurz/dataset/SUES-200-512x512
weight_save_path: /home/LVM_date/zhurz/dataset/save_model_weight

# apply LPN and set block number
LPN : 1
block : 2

# super parameters
batch_size : 8
num_epochs : 40
drop_rate : 0.35
weight_decay : 0.0001
lr : 0.01

#intial parameters
height : 150
query : drone
image_size: 384
fp16 : 0
classes : 120

model : MBF
name: MBF
```


### Test and evaluate (University-1652 Dataset)
```shell
python U1652_test_and_evaluate.py --cfg "settings.yaml" --name "your_weight_dirname_1652_2022-11-16-15:14:14" --seq 1
```

### Test and evaluate (SUES-200 Dataset)
```shell
python test_and_evaluate.py --cfg "settings.yaml" --name "your_weight_dirname_1652_2022-11-16-15:14:14" --seq 1
```


### Multiply Queries (University-1652 Dataset)
```shell
python multi_test_and_evaluate.py --cfg "settings.yaml" --multi 1 --weight "your_weight_path.pth" --csv_save_path "./result"

```

### Shifted Query (University-1652 Dataset)
```shell
python Shifted_test_and_evaluate.py --cfg "settings.yaml" --query "drone" --weight "your_weight_path.pth" --csv_save_path "./result" --gap 10
```

### Best Weights
Please check the Release page
Best weights for University-1652 Dataset have been uploaded

Any questions or suggestions feel free to contact me 
email : rzzhu24@m.fudan.edu.cn

## Relevant research

SUES-200 https://github.com/Reza-Zhu/SUES-200-Benchmark

University-1652 https://github.com/layumi/University1652-Baseline

LPN https://github.com/wtyhub/LPN

FRSA https://github.com/dmmm1997/fsra

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
