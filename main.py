import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))


'''
Using a Kaggle Dataset which contains images of benign and malignant melanoma detection.
Total samples: 13,900 count
Image dimensions: 224 x 224 pixels
''' 

df = '/kaggle/input/melanoma-cancer-dataset'

# Preparing data and create training, validation, and test inputs and labels
# label = "Benign" or "Malignant"
X_train, X_val, y_train, y_val = model_selection.train_test_split(data, labels, test_size=0.2, random_state=1)


'''
Using a KNN (K-nearest neighbors) AI model to evaluate the images and do binary classification, eventually sorting
the image into 2 categories: benign or malignant. 
'''

# Initialize our model
knn_model = KNeighborsClassifier(n_neighbors = 4)

# Train our model
knn_model.fit(X_train, y_train)

# Test our model for validation data.
y_pred = knn_model.predict(X_test)

# Print the score on the validation data.
accuracy = accuracy_score(y_test, y_pred)
print (accuracy)


'''
NOTE: This code will be integrated with the SkinGuard app, where:
1) Users will upload the photo
2) The AI model will predict whether or not the photo depicts a benign or malignant case.
3) Results will be returned to the user on the app page.

'''
