# CovPoolFER

There are following main parts:
* Covariance Pooling of Convolution Features (training and classification code is added (models to be added eventually ...))
* Temporal Pooling of Features (complete)
* Utilities for aligning SFEW (to be added ...)

#### Pooling Convolution Features
* Training and evaluation code is added. Pre-trained model to be added soon.
* [models](https://www.dropbox.com/sh/viiaeryi4nve00b/AACfZjIdpiyFYwOyUf6YskNsa?dl=0)

#### Pooling Temporal Features:
Features extracted with CNN (model proposed in the paper) from AFEW dataset are placed in zip afew_features.zip. Extract the zip to afew_features in same folder. To classify result, simple run bash classify.sh

#### Requirements
* python 2.7
* tensorflow
* numpy

#### More to come evantually ...

#### Some Notes:
* This code framework is mostly based on [facenet](https://github.com/davidsandberg/facenet)
* Apply the patch suggested in tensorflow_patch.txt file. While computing gradient of eigen-decomposition, NaNs are returned and training/inference throws errors. As suggested in file, add two lines to replace NaNs with zeros.
