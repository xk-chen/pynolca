'''
Codes of Section Illustrative Examples
'''
###############################Example##################################
from sklearn import datasets
from pynolca.nolca import NOLCA
from pynolca import kernel
from pynolca import loss
from pynolca import preprocessing
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2019)
# Load data and split
data = datasets.load_digits()
X, y = preprocessing.shuffle(data.data, data.target)
demo_kernel = kernel.Polynomial_Kernel(scale_factor = 1, intercept = 1,
                                       degree = 3)
demo_loss = loss.Ramp(parameter = -0.1, policy = "static")
# Construct the classifier
clf = NOLCA(demo_kernel, demo_loss)
# Train the classifier
clf.training(X[0:len(y) // 2], y[0:len(y) // 2], learning_rate = 0.01, reg_coefficient = 0)
print clf.predicting(X[len(y) // 2 + 1])
# Visualization
clf.plot_accuracy_curve()
clf.plot_confusion_matrix()


