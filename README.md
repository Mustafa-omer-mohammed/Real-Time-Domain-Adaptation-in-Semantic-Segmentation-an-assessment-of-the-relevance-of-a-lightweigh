# Real Time Domain Adaptation in Semantic Segmentation : An assessment of the relevance of a lightweight discriminator

### This project is part of The Machine Learning & Deep Learing cource (MLDL 2021 ) , Politecnico Di Torino , Master Of Data Science & Engineering Program 
#### The master repository https://github.com/LorenzoOttino/BiseNetv1.git


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
