import nltk
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from collections import Counter
from nltk import NaiveBayesClassifier, classify

def Process(data):
  lemmatizer = WordNetLemmatizer()
  return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(unicode(sentence,errors='ignore'))]
 

def Features_Extraction(text, setting):
  if setting=='bow':
   # Bow means bag-of-words
    return {word: count for word, count in Counter(Process(text)).items() if not word in stop}
  else:
    return {word: True for word in Process(text) if not word in stop}
    
features = [(Features_Extraction(email, 'bow'), label) for (email, label)
in emails]

def training_Model (Features, samples):
  Size = int(len(Features) * samples)
  training , testing = Features[:Size], Features[Size:]
  print ('Training = ' + str(len(training)) + ' emails')
  print ('Testing = ' + str(len(testing)) + ' emails')
  
classifier = NaiveBayesClassifier.train(training)

def evaluate(training, tesing, classifier):
  print ('Training Accuracy is ' + str(classify.accuracy(classifier,train_set)))
  print ('Testing Accuracy i ' + str(classify.accuracy(classifier,test_set)))





