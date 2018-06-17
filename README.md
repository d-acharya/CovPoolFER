# CovPoolFER

There are two parts:
* Pooling Temporal Features
* Covariance Pooling of Convolution Features (to be added soon)

#### Pooling Temporal Features:
Features extracted with CNN from AFEW dataset are placed in zip afew_features.zip. Extract the zip to afew_features in same folder. To classify result, simple run bash classify.sh

#### Requirements
* python 2.7
* tensorflow
* numpy

#### More to come soon ...

#### Some Notes:
* This code framework is mostly based on [facenet](https://github.com/davidsandberg/facenet)
* Apply the patch suggested in tensorflow_patch.txt file.
* I got some queries regarding accuracy of RAF-DB. The accuracy reported is total accuracy and authors report average of per-class accuracy. These are two different measures.

