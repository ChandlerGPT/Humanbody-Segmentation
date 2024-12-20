# 		README

## A) Create Virtual Environment

The main steps are to follow the Human3D reademe.md file. However, some adjustments were made during the process according to the actual situation of the equipment and the operation needs of the project:

Because source code compilation and installation are required, I chose the Linux system for installation (ubuntu20.04)A)  

Install Hydra and hydra-core

This installation requires attention to the corresponding python version. The python version with the highest dependency between these two is version 3.7. Therefore, when creating a virtual environment, always python==3.7. If the version is higher, an error will be reported when running the program later.

pip install python-hydra

Pip install hydra-core==1.0.5  


## B) Installation of detectron2.

The installation can be successful by following the following command, but if it is installed several times, the installation may not be successful. Therefore, the command has been distributed, and the online download and installation method has been changed to first download to the local and then install from the local.

pip3 install 

'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python   fvcore

2）pip install Cython

3）pip install git+https://github.com/philferriere/cocoapi.git

Then use cmd:

1）git clone 

git checkout 710e7795d0eeadf9def0e7ef957eea13532e34cf

cd detectron2

pip install -e . 

## C ) MinkowskiEnginede 

1）git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"

2）cd MinkowskiEngine

3）git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228

4）python setup.py install --force_cuda --blas=openblas


Other version see Human3d.yaml



## D) Run the code

1）Dataset :

https://human-3d.github.io/dataset/
![image](https://github.com/user-attachments/assets/cbdd54bc-3340-4d1f-8a33-54736495a9d7)



And withLabel dataset：

https://drive.google.com/drive/folders/1QtNufGOSBdmBeZw1o7vRUpzOZBA7d3cD?usp=sharing

download model

Run ~/Human3D/download_checkpoints.sh

URL1="https://omnomnom.vision.rwth-aachen.de/data/human3d/checkpoints/mask3d.ckpt"

URL2="https://omnomnom.vision.rwth-aachen.de/data/human3d/checkpoints/human3d.ckpt"

Save in checkpointfolder：
A）human3d.ckpt

Mask3d.ckpt


Debug
According to the results of the data set, adjust the address where the data is located when the program is running. The new address spliced ​​after reading the data in the running discovery program is different from the original address and needs to be corrected:



4）Run

Python datasets/preprocessing/humanseg_preprocessing.py preprocess --data_dir="/gemini/data-1"  --save_dir="./data/processed/egobody"  --dataset="egobody"


*Data_dir is the address where the downloaded data set is stored, Save_dir is the address where the save is run, and dataset is the database name used.


main.py

parameter setting

see script/train folder。


The setting of the parameters in the red box corresponds to the setting using egobody data. Black font is the name of the corresponding folder. The blue font is the parameter name in the file in this folder. Please read each document accordingly and will not repeat them here.

The running results are as follows:

