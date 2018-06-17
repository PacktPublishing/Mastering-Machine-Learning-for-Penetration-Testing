import numpy as np
from sklearn import *
from sklearn.linear_model import LogisticRegression

Data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)
features = Data[:,:-1]
results = Data[:, -1]
train_features = features[:2000]
test_features = features[2000:]
train_results = results[:2000]
test_results = results[2000:]
classifier = LogisticRegression()
classifier.fit(train_features, train_results)
predictions = classifier.predict(test_results)
accuracy = 100.0 * accuracy_score(test_results, predictions)
print ("The accuracy of the Phishing detector Model with Logistic Regression is: " + str(accuracy))


