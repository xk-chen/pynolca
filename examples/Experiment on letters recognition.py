from pynolca.nolca import NOLCA
from pynolca import kernel 
from pynolca import loss
from pynolca import preprocessing
import numpy as np
from time import time

# Load design matrix
X = np.loadtxt("letter-recognition.data",
                  usecols = (range(1,16)),
                  delimiter=",")
# Load the true labels
y = np.loadtxt("letter-recognition.data",
                  usecols = (0,),
                  delimiter=",", dtype='str')
# Convert true labels to valid labels
y, dictionary = preprocessing.encoder(y)
accuracies = []
num_support_vectors = []
running_time = []
parameters = [-10 ** -3, -5 * 10 ** -4, -10 ** -4, -5 * 10 ** -5, -10 ** -5, -5 * 10 ** -6, -10 ** -6, -5 * 10 ** -7, -10 ** -7, -5 * 10 ** -8]
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
    
