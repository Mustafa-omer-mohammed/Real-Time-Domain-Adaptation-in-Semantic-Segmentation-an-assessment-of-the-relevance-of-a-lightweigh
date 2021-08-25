# Real Time Domain Adaptation in Semantic Segmentation : An assessment of the relevance of a lightweight domain discriminator

### This project is part of The Machine Learning & Deep Learing cource (MLDL 2021 ) , Politecnico Di Torino , Master Of Data Science & Engineering Program 

# Abstract 
In deep CNN based models for semantic segmentation, high accuracy relies on rich spatial context (large receptive fields) and fine spatial details (high resolution), both of which incur high computational costs. The other limitation is the fact that these models are based on supervision with pixel-level ground truth but may not generalize well to unseen image domains. Our goal is to create a model  that generalize well and can adapt source ground truth labels to target domain labels while maintaining both lower computations and inference time. In our paper we combined a real-time semantic segmentation model with an unsupervised adversarial domain adaptation approach, and we used a domain classifier to align the features extracted from the output space. As the labeling process is tedious and labor intensive, we adopted a synthetic-to-real scenario. To meet our objective, we researched the effect of using a light weight discriminator network to enhance the time and avoid over-fitting by reducing the capacity of the model. We compared three different discriminator architectures and we accomplished significant enhancement both in terms of speed and segmentation performance.

# Implementation Details :
- The Semantic Segmentation Model is based on BiSeNet, pytorch 0.4.1 and python 3.6
- All the experiments were run on Google Colab Tesla T4 15 GB GPUs
- The Model contatins 3 Discriminator network and can be switchied between them by changing the parameter in the ' main Param ' ['--discrim', 'DW'  # choose Discriminator Network            DepthWise (DW) , Fully Convolutional (FC) or Fully Connected + Dropout (DR)]

# Data Sets : 
- Download CamVid dataset from [https://drive.google.com/file/d/1CKtkLRVU4tGbqLSyFEtJMoZV2ZZ2KDeA/view?usp=sharing](url)
- Download IDDA dataset from [https://drive.google.com/file/d/1GiUjXp1YBvnJjAf1un07hdHFUrchARa0/view?usp=sharing](url)
- Note: classes_info.json file needs to be modified by changing the first couple of brakets '[]' to {} and deleting the last comma.

# Segmentation Training (BaseLine Trainig) :
  python segmentation_train.py

# Adversarial Training :
  python adversarial_train.py
  
# Test :  
  python test.py
# Project Paper :
Report_Ottino_Elshaigi_Talakoobi.pdf
