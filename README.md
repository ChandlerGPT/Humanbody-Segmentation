## Point Cloud Human Body Segmentation

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

## Running Results

The running results can be reviewed after following the setup above.

![Results Screenshot](https://github.com/user-attachments/assets/121a139d-29e8-4941-9359-20c7f863ac56)
