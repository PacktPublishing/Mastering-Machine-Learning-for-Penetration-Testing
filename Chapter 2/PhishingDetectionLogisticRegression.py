import numpy as np
from sklearn import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

training_data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)
inputs = training_data[:,:-1]
outputs = training_data[:, -1]

training_inputs = inputs[:2000] 
training_outputs = outputs[:2000] 
testing_inputs = inputs[2000:] 
testing_outputs = outputs[2000:]



classifier = LogisticRegression()
classifier.fit(training_inputs, training_outputs)
predictions = classifier.predict(testing_inputs)
accuracy = 100.0 * accuracy_score(testing_outputs, predictions)
print ("The accuracy of your Logistic Regression on testing data is: " + str(accuracy))



