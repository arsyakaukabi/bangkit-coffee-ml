<p align="center">
  <img width="500" alt="logo" src="Images\logo2.svg"/>
</p>

[![](https://img.shields.io/badge/ID%20Team-C22_PC377-blue)](https://github.com/xrizer/Co-ffee)
[![PyPI version](https://badge.fury.io/py/autokeras.svg)](https://badge.fury.io/py/autokeras)
![Python](https://img.shields.io/badge/python-v3.9.0+-success.svg)
![Tensorflow](https://img.shields.io/badge/tensorflow-v2.8.0+-success.svg)


#
Automated Diagnosis of Diverse Coffee Leaf Images through a Triple Deep Convolutional Neural Network
#

## DATASETS ##

<p>The dataset used for this work came from the following works:</p>

**Please cite or credit their work when using it!** 

### **RoCoLe** 
<p>Parraga-Alava, Jorge; Cusme, Kevin; Loor, Angélica; Santander, Esneider (2019), 
<b>“RoCoLe: A robusta coffee leaf images dataset ”</b>
<i>Mendeley Data</i>, V2, doi: <a target=_blank href="http://dx.doi.org/10.17632/c5yvn32dzg.2">10.17632/c5yvn32dzg.2</a></p>

Inclusion: 
- Healthy
- Coffee Leaf Rust (CLR)
- Red Spider Mites (RSM) 

### **BrACoL** 
<p>Krohling, Renato; esgario, José; Ventura, Jose A. (2019),
<b>“BRACOL - A Brazilian Arabica Coffee Leaf images dataset to identification and quantification of coffee diseases and pests”</b>
<i>Mendeley Data</i>, V1, doi: <a target=_blank href="http://dx.doi.org/10.17632/yy2k5y8mxg.1">10.17632/yy2k5y8mxg.1</a></p>

<p>Esgario, J. G., Krohling, R. A., & Ventura, J. A. (2020) 
<b>"Deep learning for classification and severity estimation of coffee leaf biotic stress"</b>
<i>Computers and Electronics in Agriculture</i>
169, 105162. doi:<a href="https://doi.org/10.1016/j.compag.2019.105162">10.1016/j.compag.2019.105162</a></p>

Inclusion: 
- Healthy
- Coffee Leaf Rust (CLR)
- Cercospora Leaf Spots (CLS)
- Phoma Leaf Spots (PLS)
- Coffee Leaf Miner (CLM)

### **LiCoLe**
<p>Montalbo, Francis Jesmar Perez; Hernandez, Alexander Arsenio (2020) 
<b>"Classifying Barako coffee leaf diseases using deep convolutional models"</b>
<i>International Journal of Advances in Intelligent Informatics (IJAIN)</i>
[S.l.], v. 6, n. 2, p. 197-209, july 2020. ISSN 2548-3161. doi: <a href="https://doi.org/10.26555/ijain.v6i2.495">10.26555/ijain.v6i2.495</a></p>

<p>Montalbo, Francis Jesmar Perez
<b>"An Optimized Classification Model for Coffea Liberica Disease using Deep Convolutional Neural Networks"</b>
<i>n Proc. of the 2020 16th IEEE International Colloquium on Signal Processing & Its Applications (CSPA),</i> 
  Langkawi, Malaysia, 2020, pp. 213-218, doi: <a href="https://ieeexplore.ieee.org/document/9068683">10.1109/CSPA48992.2020.9068683</a>.</p>

Inclusion: 
- Healthy
- Coffee Leaf Rust (CLR)
- Sooty Molds (SM)

**For the readily prepared dataset used in this work refer to this link (OPTIONAL) : <a target=blank_ href="https://drive.google.com/drive/folders/1-CE_k_GMds2kOJDB-WfG_CCh3JN3w_ZI?usp=sharing">Google Drive Prepared Dataset<a/>** 
  
`PREPARED DATASET: (7 GB)`

***NOTE: The following credits for the datasets still goes to their appropriate owners and collectors.*** 
***please remember to cite their work when using their respective datasets.***
#

## Environment Setup

***:heavy_exclamation_mark: Make sure to create a new virtual environment preferably in Anaconda. Use Python 3.5.***


**:warning: The SWAT-DCNN uses the tensorflow GPU. This may also require at least CUDA 10 and a cuDNN**

Activate and access the folder `Co-ffee.A/` with the included `requirements.txt` file.


:warning: https://developer.nvidia.com/cuda-toolkit

https://developer.nvidia.com/rdp/cudnn-archive

Afterwards, simply enter the command in the conda CLI `pip install -r requirements.txt`

Dependencies include: 
- efficientnet==1.1.1
- colorama==0.4.3
- jupyter==1.0.0
- keras==2.2.5
- matplotlib
- numpy==1.16.2
- opencv-python==4.4.0.42
- pandas==0.25.3
- Pillow==7.2.0
- scikit-learn
- scikit-image
- scikit-plot
- scipy
- tensorflow-gpu==1.14.0

Once installed, you may either train the models individually with the `.ipynb` notebooks found in `Co-ffee.A/models/` inside the `stage-1`, `stage-2`, and `stage-3` folders or make use of the pre-trained weights.

The `Co-ffee.A/models/tdcnn/` files does not need to re-train. However, its a must to compile and aggregate the T-DCNN stages to produce its own respective weights needed by the entire SWAT-DCNN model.

## Pre-trained Weights ##

<p>The pre-trained weights are the plug and play weights that can be used to skip the training and compilation of models for the SWAT-DCNN (RECOMMENDED). :relaxed:</p>

**For an immediate simulation without the hassle of going over the previous instructions, refer to this link. : <a href="https://drive.google.com/file/d/1JNvYlat8mmpNyd3sS_QgQ5zVVEImFFjc/view?usp=sharing">Pre-Trained Weights</a>**

`PRE-TRAINED WEIGHTS FILESIZE: (271 MB)`

The filenames must not be changed for the `.h5` files.

- `model1.h5`
- `model2.h5`
- `model3.h5`

Make sure to extract the pre-trained weights in the given manner 🠊 `Co-ffee.A/weights/`

## How to use 

**heavy_exclamation_mark: Training with the pre-trained weights (RECOMMENDED)**

**heavy_exclamation_mark: Training from scratch (May take long hours depending on your PC specs) :hand:**

**:warning: Training from scratch (May take long hours depending on your PC specs) :hand:**

1. Activate your created virtual environment and enter the main `Co-ffee_A/` folder.

2. Save the dataset folder downloaded from LINK inside the `Co-ffee_A/` as `Co-ffee_A/dataset/`

3. Open the `.ipynb` files from the `Co-ffee_A/models` folder and run the following in your preferred order. The `Co-ffee_A/models/tdcnn/` is saved for later.

4. Once all models from stage-1 to 3 are trained. You may now open the `Co-ffee_A/models/tdcnn/` folder to build the T-DCNN models.

5. After all T-DCNN models are built, you may now run the `testing.py` from the main `Co-ffee_A/` folder.

6. Follow through the given instructions and make sure to use the test sample from the provided `/test/` folder

**In case of any problems, don't hesitate to contact me. I'll be happy to help.** 

## Performance Results ##

<p align="center">
  <img width="900" alt="logo" src="History\Model_History.svg"/>
</p>


