import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from matplotlib import pyplot
import numpy as np
from gensim.models import Word2Vec
import gensim.models
from sklearn.feature_extraction.text import CountVectorizer

# IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,valid_portion=0.1)

trainX, trainY = train
testX, testY = test

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

# Network building
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
ourmodel = tflearn.DNN(net, tensorboard_verbose=0)
ourmodel.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=32)

#accuracy
predictions = (np.array(ourmodel.predict(testX))[:,0] >= 0.5).astype(np.int_)
test_accuracy = np.mean(predictions == testY[:,0], axis=0)
print("Test accuracy: ", test_accuracy)



from sklearn.feature_extraction.text import HashingVectorizer
# list of text documents
text = ["i am angry and sad too"]
# create the transform
vectorizer = HashingVectorizer(n_features=100)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(vector.toarray())
positive_prob = ourmodel.predict(vector.toarray().reshape(1,100))
print(positive_prob)

text = ["I am happy and i love it"]
# create the transform
vectorizer = HashingVectorizer(n_features=100)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(vector.toarray())
positive_prob = ourmodel.predict(vector.toarray().reshape(1,100))
print(positive_prob)

#print('P(positive) = {:f} :'.format(positive_prob),'Positive' if positive_prob > 0.5 else 'Negative')