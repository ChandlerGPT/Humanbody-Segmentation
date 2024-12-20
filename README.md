# Point Cloud Human Body Segmentation
This repo is implementation of 5 models for point cloud segmentation
The paper of these models are:
[PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)
[PointNet++](https://proceedings.neurips.cc/paper_files/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf)
[Point Transformer](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf)
[Kernel Point Convolution](https://openaccess.thecvf.com/content_ICCV_2019/papers/Thomas_KPConv_Flexible_and_Deformable_Convolution_for_Point_Clouds_ICCV_2019_paper.pdf)
[Straitified Transformer](https://openaccess.thecvf.com/content/CVPR2022/papers/Lai_Stratified_Transformer_for_3D_Point_Cloud_Segmentation_CVPR_2022_paper.pdf)
[Human3D](https://human-3d.github.io/assets/Human3D_paper.pdf)
## PointNet
**Relevant Files(under `pointnet_pointnet+`):**
### Install
The latest codes are tested on Ubuntu 16.04, CUDA10.1, PyTorch 1.6 and Python 3.7:
```shell
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
```
### Data Preparation
- Download the THuman dataset and save in `data/thuman`.
- `data_utils/human_util.py`
- `data_utils/THumanDataLoader.py`
### Run
- Training file is in `train_semseg_humanbody.py`.
- Testing file is in `test_semseg.py`
```shell
python train_semseg_humanbody.py --model pointnet_sem_seg
```
## PointNet++
**Relevant Files(under `pointnet_pointnet+`):**
### Data Preparation
- Download the THuman dataset and save in `data/thuman`.
- `data_utils/human_util.py`
- `data_utils/THumanDataLoader.py`
### Run
- Training file is in `train_semseg_humanbody_pointnet2.py`.
- Testing file is in `test_semseg.py`
```shell
python train_semseg_humanbody_pointnet2.py --model pointnet2_sem_seg
```
## Point Transformer
**Relevant Files(under `point-transformer-main`):**
- Configuration file: `config/s3dis/thuman_pointtransformer_repro.yaml`
### Data Preparation
- Download the THuman dataset and save in `data/thuman`
- `util/thuman.py`
- Label: `data/thuman_names.txt`
- Download the THuman dataset and save in `data/thuman`.
### Run
- Training file is in `tool/train.py`
- Testing file is in `tool/test.py`
- Train the model
```shell
 sh tool/train.sh thuman pointtransformer_repro
```
- Test the model Specify the gpu
```shell
CUDA_VISIBLE_DEVICES=0 sh tool/test.sh thuman pointtransformer_repro
```
## Experimental Results

- Semanctic Segmentation with PointNet PointNet++ PointTransformer

  |Model | mAcc |mIoU |
  |-------| ------| -------|
  |PointNet| 80.2 | 42.6 |
  |PointNet++| 85.6 | 52.3 |
  |Point Transformer|88.1|69.3|
---


## Stratified Transformer
**Relevant Files (under `Stratified-Transformer-main`):**

- **Dataset & Data Preparation:**
  - `util/HumanBodySegmentation.py`
  - `util/data_util.py` (including `data_prepare_humanbody` and other data preparation functions)
  
- **Training Code:**
  - `train_humanAdded.py`
  
- **Configuration File:**
  - `config/HumanBodySegmentation/HumanBodySegmentation_stratified_transformer.yaml`
  
- **Testing Code:**
  - `test_humanAdded.py`

**Note:** Use the same configuration file as in training.
## Kernel Point Convolution
**Relevant Files (under `KPConv-master`):**

- `training_HumanBodySegmentationDataset.py`: Main training script for KPConv on the HumanBodySegmentation dataset.
- `HumanBodySegmentation.py`: Dataset loading and preprocessing, including point cloud reading, subsampling, and batch creation.
- `models/KPFCNN_model.py`: KPConv model definition, implementing kernel point convolutions and network architecture.
- `utils/config.py`: General configuration for adjusting dataset and task parameters.
- `utils/trainer.py`: Trainer handling training loops, validation, logging, and checkpointing.
## human3d reprodcution
### A) Create Virtual Environment

The main steps are to follow the Human3D `reademe.md` file. However, some adjustments were made during the process according to the actual situation of the equipment and the operation needs of the project:

Because source code compilation and installation are required, the Linux system was chosen for installation (Ubuntu 20.04).

#### Install Hydra and hydra-core

This installation requires attention to the corresponding Python version. The highest dependency for these two libraries is Python 3.7. Therefore, when creating a virtual environment, always use `python==3.7`. If a higher version is used, an error will be reported when running the program later.

```bash
pip install python-hydra
pip install hydra-core==1.0.5
```

### B) Installation of Detectron2

The installation can be completed using the following commands. However, if the installation fails multiple times, it may be necessary to use local installation by downloading the package first.

```bash
pip3 install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python fvcore
pip install Cython
pip install git+https://github.com/philferriere/cocoapi.git
```

Using the terminal:

```bash
git clone https://github.com/facebookresearch/detectron2.git
git checkout 710e7795d0eeadf9def0e7ef957eea13532e34cf
cd detectron2
pip install -e .
```

### C) Installation of MinkowskiEngine

```bash
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas
```

Other versions can be configured in `Human3D.yaml`.

### D) Run the Code

#### Dataset

- Download dataset from: [Human3D Dataset](https://human-3d.github.io/dataset/)

![Dataset Screenshot](https://github.com/user-attachments/assets/cbdd54bc-3340-4d1f-8a33-54736495a9d7)

- With labeled dataset: [WithLabel Dataset](https://drive.google.com/drive/folders/1QtNufGOSBdmBeZw1o7vRUpzOZBA7d3cD?usp=sharing)

#### Download Model

Run the following script to download the checkpoints:

```bash
~/Human3D/download_checkpoints.sh
```

URLs for the models:

```bash
URL1="https://omnomnom.vision.rwth-aachen.de/data/human3d/checkpoints/mask3d.ckpt"
URL2="https://omnomnom.vision.rwth-aachen.de/data/human3d/checkpoints/human3d.ckpt"
```

Save in the `checkpoint` folder:
- `human3d.ckpt`
- `mask3d.ckpt`

![Model Screenshot](https://github.com/user-attachments/assets/ed103666-a5f2-4e02-b964-8ed4b74b0d34)

#### Debug

Adjust the address of the dataset as needed based on the program's requirements. The newly generated address may differ from the original and needs to be corrected.

![Debug Screenshot](https://github.com/user-attachments/assets/735bc469-7f2b-4fe8-bc76-a51e599b68cb)

#### Run Preprocessing Script

```bash
python datasets/preprocessing/humanseg_preprocessing.py preprocess --data_dir="/gemini/data-1" --save_dir="./data/processed/egobody" --dataset="egobody"
```

![Preprocessing Screenshot](https://github.com/user-attachments/assets/f35729ec-38a2-4241-9f0b-757629d78c82)

- `data_dir`: Address where the downloaded dataset is stored.
- `save_dir`: Address to save processed data.
- `dataset`: Database name being used.

#### Main Program

For parameter settings, refer to the `script/train` folder.

![Parameter Settings Screenshot](https://github.com/user-attachments/assets/148c6fb4-2d6e-4b86-b4c7-f33b9709a9e3)

The parameters for using the `egobody` data are highlighted as follows:
- Black font indicates folder names.
- Blue font indicates parameter names in the file within the folder.

Please read the corresponding documents for details.

---

#### Running Results

The running results can be reviewed after following the setup above.

![Results Screenshot](https://github.com/user-attachments/assets/121a139d-29e8-4941-9359-20c7f863ac56)

## Citation
```
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
```
```
@article{thomas2019KPConv,
    Author = {Thomas, Hugues and Qi, Charles R. and Deschaud, Jean-Emmanuel and Marcotegui, Beatriz and Goulette, Fran{\c{c}}ois and Guibas, Leonidas J.},
    Title = {KPConv: Flexible and Deformable Convolution for Point Clouds},
    Journal = {Proceedings of the IEEE International Conference on Computer Vision},
    Year = {2019}
}
```
```
@inproceedings{lai2022stratified,
  title={Stratified Transformer for 3D Point Cloud Segmentation},
  author={Lai, Xin and Liu, Jianhui and Jiang, Li and Wang, Liwei and Zhao, Hengshuang and Liu, Shu and Qi, Xiaojuan and Jia, Jiaya},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8500--8509},
  year={2022}
}
```
```
@inproceedings{zhao2021point,
  title={Point transformer},
  author={Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip HS and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16259--16268},
  year={2021}
}
```
```
@InProceedings{yan2021sparse,
  title={Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion},
  author={Yan, Xu and Gao, Jiantao and Li, Jie and Zhang, Ruimao, and Li, Zhen and Huang, Rui and Cui, Shuguang},
  journal={AAAI Conference on Artificial Intelligence ({AAAI})},
  year={2021}
}
```
```
@InProceedings{yan20222dpass,
      title={2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds}, 
      author={Xu Yan and Jiantao Gao and Chaoda Zheng and Chao Zheng and Ruimao Zhang and Shuguang Cui and Zhen Li},
      year={2022},
      journal={ECCV}
}
```
