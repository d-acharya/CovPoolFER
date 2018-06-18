# CovPoolFER

There are following main parts:
* Covariance Pooling of Convolution Features (partial code is added (models to be added eventually ...))
* Temporal Pooling of Features (complete)
* Utilities for aligning SFEW

#### Pooling Convolution Features
Training and evaluation code is added. Pre-trained model to be added soon.

#### Pooling Temporal Features:
Features extracted with CNN (model proposed in the paper) from AFEW dataset are placed in zip afew_features.zip. Extract the zip to afew_features in same folder. To classify result, simple run bash classify.sh

#### Requirements
* python 2.7
* tensorflow
* numpy

#### More to come evantually ...

#### Some Notes:
* This code framework is mostly based on [facenet](https://github.com/davidsandberg/facenet)
* Apply the patch suggested in tensorflow_patch.txt file.
