# Authors

* Güler Özgür, [@oezgueracar](https://github.com/oezgueracar)
* Tan Xiao, [@xiao-gh](https://github.com/xiao-gh)
* Xie Yichun, [@Y1CX](https://github.com/y1cx)

# Information

## Hate Speech Training Model based on BERT

A BERT model/pipeline that is using a dataset to identify hate speech from various social media channels (Reddit, Twitter, Youtube) and Wikipedia by Salminen et al. (2020).

The code is modified, extended and shortened according to our needs. For further information about the dataset and the model, see *References* further below.

In short, the dataset includes a column for the comment, a column whether the comment was hate speech (1) or not (0), and a column for its source (Reddit, Twitter, Wikipedia, Youtube). More details can also be found in our report.

## Pipeline Structure

The following Files are part of the training pipeline:

* 1_train_bert.ipynb
* 2_create_bert_features.ipynb
* 3_train_xgboost.ipynb
* 4_predict.ipynb
* functions_modified.py
* bert_functions_modified.py

### 1_train_bert

This python notebook will take the raw dataset and will train the base BERT model.

### 2_create_bert_features

This script will use the trianed BERT model to create BERT features on the full training dataset, which will later be used as the input of XGBoost training.

### 3_train_xgboost

With input of the BERT featuresthe, XGBoost classifier will be trained and the performance will be evaluated on the test set.

### 4_predict

By providing the trained BERT model from script 1 and the XGBoost classifier from script 3, this file will achieve the task of classifying unseen text to see whether it's hate speech or not.

### functions_modified and bert_functions_modified

These are helper functions that need to be in the working directory in order for the training and evaluations of the models to work.

# References

Salminen, J., Hopf, M., Chowdhury, S. A., Jung, S. G., Almerekhi, H., & Jansen, B. J. (2020). Developing an online hate classifier for multiple social media platforms. *Human-centric Computing and Information Sciences*, *10*(1), 1-34.

https://github.com/joolsa/Binary-Classifier-for-Online-Hate-Detection-in-Multiple-Social-Media-Platforms

# Download Link for the Models

BERT_model only: https://drive.google.com/file/d/13YAvExFu39HkIvBXtuAlcP3puh16o6Pk/view?usp=sharing

XGboosted BERT model: https://drive.google.com/file/d/1UqvXRzkXMvgNPx_sCQy6dkQRsC4ZMBOi/view?usp=sharing
