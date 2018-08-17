# CovPoolFER

There are following main parts:
* Covariance Pooling of Convolution Features
* Temporal Pooling of Features

#### Pooling Convolution Features
You can download following models (2.5 GB total)
* [models](https://drive.google.com/open?id=1SmFPoX3ASqUXbvtOYFGJnMzr9PhHHjZq)
and run reproducepaper.sh (after uncommenting appropriate lines)
* For the code for inception-resnet-v1, I the used same implementation of inception-resnet in [facenet](https://github.com/davidsandberg/facenet)
* For baseline, the network is same as included here except this code contains few additional (covariance pooling) layers.

#### Pooling Temporal Features:
Features extracted with CNN (model proposed in the paper) from AFEW dataset are placed in zip afew_features.zip. Extract the zip to afew_features in same folder. To classify result, simple run bash classify.sh

#### Requirements
* python 2.7
* tensorflow
* numpy
* sklearn

#### Some Notes:
* This code framework is mostly based on [facenet](https://github.com/davidsandberg/facenet)
* Apply the patch suggested in tensorflow_patch.txt file. While computing gradient of eigen-decomposition, NaNs are returned by tensorflow when eigenvalues are identical. This throws error and cannot continue training. The patch only replaces NANs with zeros. This makes training easier.
