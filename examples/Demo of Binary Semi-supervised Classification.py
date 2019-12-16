import numpy as np
from pynolca import preprocessing
from sklearn import datasets
from pynolca.nolca import NOLCA
# Load data
np.random.seed(2019)
data = datasets.load_breast_cancer()
X, y = preprocessing.shuffle(data.data, data.target)
# Wipe 50 percent labels
unlabelled_proportion = 0.5
unlabelled = np.random.choice(len(y), int(unlabelled_proportion * len(y)))
y[unlabelled] = -1
# Construct the classifier
clf = NOLCA()
# Train the classifier
clf.training(X[: -1], y[: -1], learning_rate = 0.01, 
	     reg_coefficient = 0.1, unlabelled = True)
# Visualization
clf.plot_accuracy_curve()
clf.plot_confusion_matrix()
