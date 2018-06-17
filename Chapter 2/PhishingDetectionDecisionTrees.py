from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np

data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)
features = data[:,:-1]
results = data[:, -1]

training_features = features[:2000]
training_results = results[:2000]
testing_features = features[2000:]
testing_results = results[2000:]
classifier = tree.DecisionTreeClassifier()
classifier.fit(training_features, training_results)
predictions = classifier.predict(testing_features)
accuracy = 100.0 * accuracy_score(testing_results, predictions)

print ("The accuracy of the Phishing detector Model with Decision Tree" + str(accuracy))

