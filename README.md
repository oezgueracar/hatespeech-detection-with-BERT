# Authors

* Güler Özgür, [@oezgueracar](https://github.com/oezgueracar)
* Tan Xiao, [@xiao-gh](https://github.com/xiao-gh)
* Xie Yichun, [@Y1CX](https://github.com/y1cx)

# Information

## Hate Speech Training Model based on BERT

A BERT model/pipeline that is using a dataset to identify hate speech from various social media channels (Reddit, Twitter, Youtube) and Wikipedia by Salminen et al. (2020).

The code is modified, extended and shortened according to our needs. For further information about the dataset and the model, see *References* further below.

In short, the dataset includes a column for the comment, a column whether the comment was hate speech (1) or not (0), and a column for it source (Reddit, Twitter, Wikipedia, Youtube). More details can also be found in our report.

## Pipeline Structure

The following Files are part of the training pipeline:

* 1_train_bert.ipynb
* 2_predict.ipynb
* functions.py
* bert_functions_modified.py

### 1_train_bert

This python notebook will take the raw dataset and will train the base BERT model.

### 2_predict

By providing either the base BERT model from step 1, this file will allow to use the model to classify unseen text whether it's hate speech or not.

### functions and bert_functions_modified

These are helper files that need to be in the working directory in order for the BERT training to work. In case of Colab, they need to be uploaded (along the dataset) to /content/ first.

# References

Salminen, J., Hopf, M., Chowdhury, S. A., Jung, S. G., Almerekhi, H., & Jansen, B. J. (2020). Developing an online hate classifier for multiple social media platforms. *Human-centric Computing and Information Sciences*, *10*(1), 1-34.

https://github.com/joolsa/Binary-Classifier-for-Online-Hate-Detection-in-Multiple-Social-Media-Platforms

# Download Link for the Models

BERT_model only: https://drive.google.com/file/d/13YAvExFu39HkIvBXtuAlcP3puh16o6Pk/view?usp=sharing

XGboosted BERT model: https://drive.google.com/file/d/1UqvXRzkXMvgNPx_sCQy6dkQRsC4ZMBOi/view?usp=sharing
