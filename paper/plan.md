## **TITLE(s)** ideas

Long-wave infrared sky images classification and probabilistic segmentation of cloud structures with deep-learning

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

### 2. BACKGROUND
#### 	2.1 Motivation

- StarDICE experiment
     <u>*Figure 1 schematics of StarDICE experiment used in the poster for MORIOND*</u>
- importance of good data
     <u>=> infrared monitoring of the atmosphere : flag observations</u>

#### 	2.2 Related work

### 3. EXPERIMENTAL SETUP AND DATASET

#### 	3.1 Infrared thermal imaging system

  - <u>*Figure 2 Same as Figure 1 of REFERENCE_PAPER_AMT.*</u>

#### 	3.2 Dataset and pre-processing

   - inspired from 2. of REFERENCE_PAPER_AMT

   - <u>*Figure 3 same as Fig. 8 of TransCloudSeg but add clear sky image line and synthetic clear sky image line.*</u>

   - <u>*Table same as EVALUATION.pdf Tab. 2 Overview of the collected image data set but add synthetic images and one row for clear and  a second for cloud*</u>
        - inspire from 3.1 of EVALUATION.pdf
        

### 4. IMAGE CLASSIFICATION AND CLOUD STRUCTURE DETECTION FRAMEWORK FOR INFRARED SKY IMAGES

#### 	4.1 Overall framework of XXXX

<u>*Figure 4 : same as TransCloudSeg Figure 1 OR same as in DEEPCLOUD.pdf Figure 1.*</u>

#### 	4.2 CNN architecture for image classifier

- *Figure 5 : same as Figure 3 of REFERENCE_PAPER_AMT.*

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

  - real-time data analysis module as in ACS.pdf
  - re-training on larger dataset

### 7. CONCLUSION





Acknowledgments, author contributions, etc are the same as in DIURNAL_AMT

Code and data availability : For reproducibility, the codebase and dataset used for the experiments carried out in this article is released at \url{GitHub repo}