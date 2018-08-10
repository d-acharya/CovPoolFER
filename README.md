# CovPoolFER

There are following main parts:
* Covariance Pooling of Convolution Features
* Temporal Pooling of Features

#### Pooling Convolution Features
You can download following models (2.5 GB total)
* [models](https://drive.google.com/open?id=1SmFPoX3ASqUXbvtOYFGJnMzr9PhHHjZq)
and run reproducepaper.sh (after uncommenting appropriate lines)

#### Pooling Temporal Features:
Features extracted with CNN (model proposed in the paper) from AFEW dataset are placed in zip afew_features.zip. Extract the zip to afew_features in same folder. To classify result, simple run bash classify.sh

#### Requirements
* python 2.7
* tensorflow
* numpy

#### Some Notes:
* This code framework is mostly based on [facenet](https://github.com/davidsandberg/facenet)
* Apply the patch suggested in tensorflow_patch.txt file. While computing gradient of eigen-decomposition, NaNs are returned and training/inference throws errors. As suggested in file, add two lines to replace NaNs with zeros.
