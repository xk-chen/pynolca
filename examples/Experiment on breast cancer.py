from sklearn import datasets
from pynolca.nolca import NOLCA
from pynolca import kernel 
from pynolca import loss
from pynolca import preprocessing
import numpy as np
from time import time

data = datasets.load_breast_cancer()
X, y = data.data, data.target
accuracies = []
num_support_vectors = []
running_time = []
parameters = [-1, -10 ** -13, -10 ** -14, -10 ** -15]
for i in range(len(parameters)):
    # Construct kernel and loss
    Kernel = kernel.RBF_Kernel()
    Loss = loss.Ramp(policy = "static", parameter = parameters[i])
    # Construct classifier
    clf = NOLCA(Kernel, Loss)
    # Train
    start = time()
    clf.training(X, y, learning_rate = 0.01)
    end = time()
    elapsed = end - start
    # Visualization
    num_support_vectors.append(clf.get_num_support_vectors())
    accuracies.append(clf.get_accuracy()[-1])
    running_time.append(elapsed)
print accuracies
print num_support_vectors
print running_time
