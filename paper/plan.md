## **TITLE(s)** ideas

New infrared sky image classification and cloud segmentation using deep-learning framework for ground-based radiometric camera observation

IRIS-CloudDeep: Infrared Radiometric Image classification and Segmentation of CLOUD structure using Deep-learning framework for ground-based radiometric camera observation

IRIS-CloudNet: Infrared Radiometric Image classification and Segmentation of CLOUD structure using neural Networks for ground-based radiometric camera observation

## **PLAN**

### 1. INTRODUCTION

- generalization about clouds
- ways of observing clouds
- ground-based infrared observations (give examples)
- relation to cosmological surveys
- what we do regarding this manner in the StarDICE experiment
- plan announcement + List of paper main contributions as in DEEPCLOUD.pdf

*Figure 1 : same as TransCloudSeg Figure 1 OR same as in DEEPCLOUD.pdf Figure 1.*

### 2. BACKGROUND
#### 	2.1 Motivation

- StarDICE experiment
     *Figure 2 schematics of StarDICE experiment used in the poster for MORIOND*
- importance of good data
     => infrared monitoring of the atmosphere : flag observations

#### 	2.2 Related work

##### 		2.2.1 Ground-Based sky/cloud monitoring
    	2.2.1 Ground-Based Cloud Image Segmentation

​		*Full page width Table same as Table 1 in ACLNet.pdf*

### 3. EXPERIMENTAL SETUP AND DATASET

#### 	3.1 Infrared thermal imaging system

  - *Figure 3 Same as Figure 1 of REFERENCE_PAPER_AMT.*

#### 	3.2 Dataset and pre-processing

##### 		3.2.1 Data description

   - inspired from 2. of REFERENCE_PAPER_AMT

   - *Figure 4 same as Fig. 8 of TransCloudSeg but add clear sky image line and synthetic clear sky image line.*

   - *Table same as EVALUATION.pdf Tab. 2 Overview of the collected image data set but add synthetic images and one row for clear and  a second for cloud*

        - inspire from 3.1 of EVALUATION.pdf
        

    ##### 3.1.2 Ground-truth masks
    
    ##### 3.1.3 Synthetic images
    
    ##### 3.1.4 Augmentations

### 4. IMAGE CLASSIFICATION AND CLOUD STRUCTURE DETECTION FRAMEWORK FOR INFRARED SKY IMAGES

#### 	4.1 Overall framework of XXXX

- *Figure 5 : same as Figure 2 TransCloudSeg OR same as Figure 2 in CLOUDEEPLAPV3.pdf*

#### 	4.2 CNN architecture for image classifier

- *Figure 6 : same as Figure 3 of REFERENCE_PAPER_AMT.*

##### 		4.2.1 Encoder

##### 		4.2.2 Decoder

##### 		4.2.3 Softmax classifier

#### 	4.3 U-Net model architecture for cloud segmentation

- *Figure 7 : same as Figure 3 of REFERENCE_PAPER_AMT*

##### 		4.3.1 Encoder

##### 		4.3.2 Decoder

##### 		4.3.3 Model output

#### 	4.4 Training procedure

- *Figure 8 dual plot same as ACS.pdf Fig. 5.*
- inspired from 3.2 REFERENCE_PAPER_AMT
- Loss-function : see TransCloudSeg III E
- Hyperparameters optimization with Optuna

### 5. EXPERIMENTS

#### 	5.1 Performance/evaluation metrics/criteria

- precision, accuracy, recall, F1_score, mIoU, error rate

- inspired from TransCloudSeg IV B., DIURNAL_AMT.pdf section 4.2

+ formulas from 3.2 of CLOUDEEPLAPV3.pdf

#### 	5.2 Framework effectiveness

- *Figure 9 : ROC Curve for class + seg. as in Fig 2. from 12_TESTS_NEW_DATASET_COMPARISON.pdf + add ROC from other segmentation models same as Figure 3 in ACLNet.pdf*

  ##### 5.2.2 Classifier

  ##### 5.2.3 Segmentation

#### 	5.3 Comparison with state-of-the-art methods

- *Table same as Table VI of TransCloudSeg or Table same as in Table 6 from CLOUDEEPLAPV3.pdf or same as Table 2 in ACLNet.pdf + add column Running Times [ms] for prediction as in Table IV of TransCloudSeg*
- same as 2) IV C TransCloudSeg. Insist on the non-feasibility to compare to traditional methods as they rely on color information from red and blue channels.
- inspired from IV B from 12_TESTS_NEW_DATASET_COMPARISON.pdf

### 6. DISCUSSION

#### 	6.1 Segmentation performance to publicly available datasets

- inspired from IV C from 12_TESTS_NEW_DATASET_COMPARISON.pdf
- *Table as Table VI from TransCloudSeg*

#### 	6.2 Comparison to Otsu method

- inspired from 4.2 REFERENCE_PAPER_AMT

#### 	6.3 Future perspectives

  - real-time data analysis module
  - re-training on larger dataset

### 7. CONCLUSION





Acknowledgments, author contributions, etc are the same as in DIURNAL_AMT

Code and data availability : For reproducibility, the codebase and dataset used for the experiments carried out in this article is released at \url{GitHub repo}