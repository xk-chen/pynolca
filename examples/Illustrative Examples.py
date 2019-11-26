'''
Codes of Section Illustrative Examples
'''
###############################Example_1##################################
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
num_stage_1 = len(y) // 2
X_stage_1, y_stage_1 = X[: num_stage_1], y[: num_stage_1]
X_stage_2, y_stage_2 = X[num_stage_1: ], y[num_stage_1: ]
# Construct kernel and loss
demo_kernel = kernel.Polynomial_Kernel(scale_factor = 1, intercept = 1,
                                       degree = 3)
demo_loss = loss.Ramp(parameter = -0.1, policy = "static")
# Construct the classifier
clf = NOLCA(demo_kernel, demo_loss)
# Train the classifier
clf.training(X_stage_1[: -1], y_stage_1[: -1], learning_rate = 0.01, 
				 reg_coefficient = 0)
# Visualization
clf.plot_accuracy_curve()
clf.plot_confusion_matrix()
# Predict
print clf.predicting(X_stage_1[-1])
# Show the true label
plt.imshow(X_stage_1[-1].reshape(8,8), cmap = plt.cm.hot)
plt.show()
# Retrain

clf_new = NOLCA(demo_kernel, demo_loss,
                num_support_vectors = clf.get_num_support_vectors(),
                support_vectors = clf.get_support_vectors(),
                sample_weight = clf.get_weight())

clf_new.training(X_stage_2, y_stage_2, learning_rate = 0.01,
                 reg_coefficient = 0)
# Visualization
clf_new.plot_confusion_matrix()
clf_new.plot_accuracy_curve()


##############################Example_2###################################
# Load data
np.random.seed(2019)
data = datasets.load_breast_cancer()
X, y = preprocessing.shuffle(data.data, data.target)
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
plt.show()
