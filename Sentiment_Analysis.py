import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#word cloud

# reading csv file for analysis
data=pd.read_csv("Tweets.csv")

# maksing values of Y 0=neutral , -1 for negative & 1 for positive
data["airline_sentiment"]=data.airline_sentiment.map({"neutral":"0","positive":"1","negative":"-1"})

# preprocessing
X=data.iloc[:,10]
print(X.size)
    
Y=data.iloc[:,1]

# displaying sentiment count 

sentiment=data['airline_sentiment'].value_counts()

label=['Negative','Neutral','Positive']
index = np.arange(len(label))

plt.bar(index,sentiment)
plt.xticks(index,label,rotation=45)
plt.ylabel('Sentimen Count')
plt.xlabel('Sentiment')
plt.title('Sentiment')


def plot_for_each_airline(airline_name):
        airline_data=data[data['airline']==airline_name]
        sentiment=airline_data['airline_sentiment'].value_counts()
        label=['Negative','Neutral','Positive']
        index = np.arange(len(label))
        
        plt.bar(index,sentiment)
        plt.xticks(index,label,rotation=45)
        plt.ylabel('Sentimen Count')
        plt.xlabel('Sentiment')
        plt.title(airline_name)
    
#run each line seperately    
plot_for_each_airline("Virgin America")
plot_for_each_airline("American")
plot_for_each_airline("United")
plot_for_each_airline("Southwest")
plot_for_each_airline("Delta")
plot_for_each_airline("US Airways")



# splitting training data set and test data set
from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer()
bag_of_words=vectorizer.fit_transform(X)
vc=vectorizer.get_feature_names
print(bag_of_words.toarray())
print( vectorizer.vocabulary_)

X_train,X_test,Y_train,Y_test=train_test_split(bag_of_words,Y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train.toarray(),Y_train)

predict=model.predict(X_test.toarray())
print(model.score(X_test.toarray(),Y_test))
print(model.score(bag_of_words,Y))


# working on user defined input 

def user_defined_sentences(statement):
    st=[s]
    vect=vectorizer.transform(st)
    print(vect.toarray())
    predict=model.predict(vect.toarray())
    print(predict)
    polarity=model.predict_proba(vect.toarray())
    print(polarity)

    # uncomment to see linear graph
    # plt.plot(index,polarity.reshape(-1,1))
    # plt.show()
    
    # plotting horizontal bar graph for better demonstration 
    plt.bar(index,polarity[0])
    plt.ylabel('Probability',fontsize=10)
    plt.xlabel('Prediction',fontsize=10)
    plt.xticks(index, label, fontsize=10, rotation=40)
    plt.title('Sentiment of Sentence')
    plt.show()
    
s="Not happy with the flight, too boring and late"
user_defined_sentences(s)









