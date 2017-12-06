import numpy as np
import pandas as pd
import nltk
import re
import os
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
 
    
# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def extract_features(size_data,imp_words,given_data):
    features_matrix = np.zeros(size_data)
    add_data = given_data["additionalAttributes"]
    # extracting features from words extracted from additional attribute
    i=0;
    for sample_word in imp_words:
        k=0;
        for row in add_data:
            if sample_word[0] in str(row).lower():
                features_matrix[k,i]=1
            k=k+1
        i=i+1;
    
    bread_data = given_data["breadcrumbs"]

    # features from breadcrumb for book
    i=0;
    for row in bread_data:
        bd= str(row).split(">")
        k=0;
        for word in bd:
            if "book" in word.lower():
                features_matrix[i,-3]=k+1
                k=k+1
        i=i+1

    # features from breadcrumb for music
    i=0;
    for row in bread_data:
        bd= str(row).split(">")
        k=0;
        for word in bd:
            if "music" in word.lower():
                features_matrix[i,-2]=k+1
                k=k+1
        i=i+1

    #features from breadcrumb for video
    i=0;
    for row in bread_data:
        bd= str(row).split(">")
        k=0;
        for word in bd:
            if "video" in word.lower():
                features_matrix[i,-1]=k+1
                k=k+1
        i=i+1

    return features_matrix


#Reading data from training file given
data = pd.read_csv("train.csv")
add_data =data["additionalAttributes"]

newdict={};  # to make dictionary of all key value pairs in additional attrinutes given

for row in add_data[~add_data.isnull()]:
    x=row.split(";")
    ds=dict(word.split('=',1) for word in x if "=" in word)
    newdict=dict(newdict,**ds)


# to filter/clean out keys of dictionarie which will be later used in extracting features dictionaries
# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')

# load nltk's SnowballStemmer as variabled 'stemmer'
stemmer = SnowballStemmer("english")

# Now I am trying to build a vocabulary of important words from the keys of additional attributes
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in newdict.keys():
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'keys' of attribute pairs, tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
#print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
#print (vocab_frame.head())

# remove stopwords
filtered_vocab = [w for w in vocab_frame["words"] if not w in stopwords]
# count the frequency of words which are most common
n_words=40 # Number of features from additional attribute, empirically I have taken 40. can set it to another number
x = Counter(filtered_vocab)
imp_words=x.most_common(n_words)
size_frame=(len(data),n_words+3)
z_data= extract_features(size_frame,imp_words,data)
# stroing in dataframe
c_names= [item[0] for item in imp_words]
c_names=c_names + ["bread_book","bread_music","bread_video"]
dd_train=pd.DataFrame(z_data,columns=c_names)


######## features extraction from training data done! ############### 

#Naive Bayes classifier making
X_train = z_data
y_train= data["label"]

# extracting features from test data
test_data = pd.read_csv("evaluation.csv")
size_test=(len(test_data),n_words+3)
X_test= extract_features(size_test,imp_words,test_data)
c_names= [item[0] for item in imp_words]
c_names=c_names + ["bread_book","bread_music","bread_video"]
dd_test=pd.DataFrame(X_test,columns=c_names)
 
# training a Naive Bayes classifier

gnb = GaussianNB().fit(X_train, y_train)
gnb_predictions = gnb.predict(X_test)
new_column = pd.DataFrame({'label': gnb_predictions}) 
final_data=pd.concat([test_data, new_column], axis=1)
final_data.to_csv('submissions.csv') 
print('Submission file genertaed successfully')
