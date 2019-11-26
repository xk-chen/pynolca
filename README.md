# pynolca: A Noise-resilient Online Large-scale Classification Algorithm Package in Python
**Copyright (c) 2018-2019, SMaLL. All rights reserved.**

#Introduction 
**pynolca** is a computationally efficient classification package in Python 2.7.x. with noise-resilient functionality. The main advantage of **pynolca** is that it is not only able to deal with conventional classification and binary semi-supervised classification tasks in an online learning fashion efficiently but also suitable for noisy scenarios with the help of noise-resilient loss function. Besides, the kernel module of this package allows users to construct complex, non-linear models, which can promote the classification accuracy significantly. To ease the burden of users, the package provides preprocessing module to tackle some typical problems independent of classifications.


##Dependencies: 
|  Packages   | Version  |
|  ----  | ----  |
| NumPy  | >= 1.8.2 |
| Matplotlib  | >= 2.0.0|
|scikit-learn| >= 0.18.1 (optional, for examples)|

##Installation: 

1. Download the release of pynolca from
  [https://github.com/xk-chen/pynolca].
2. Unzip the file and execute: `
$ python setup.py install --user
` in the root directory of pynolca.
