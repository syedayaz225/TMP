# Importing essential libraries
import pandas as pd
import sklearn
from pandas import DataFrame
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, average_precision_score
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
from collections import Counter
import matplotlib.pyplot as plt
#Import Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.porter import PorterStemmer
import csv
import nltk 
from nltk.corpus import brown
#from nltk.probability import ConditionalFreqDist

def lemmatize(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word=word)

def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word=word)


def sents(path):
    """
        PRE-PROCESSING DATA
        get data
        -------
        Data Frame (Rectangular Grids)
        DataFrames make manipulating your data easy, from selecting or replacing columns and indices to
        reshaping your data.
        -------
    """

    data = pd.read_csv( path , sep = "\t", index_col=False, encoding='latin-1', low_memory=False)
    df = DataFrame(data)
#     print(df['Sentiment'])
    labelCount = df.groupby(df['Sentiment']).count()
    #print(labelCount)
    x = df['SentimentText'].str.replace('http\S+|www.\S+', '', case=False)
    y = df['Sentiment']
    x = x.str.replace('[^a-zA-Z]', ' ') #
    x_check = [" ".join([lemmatize(word) for word in sentence.split(" ")]) for sentence in x]
    stopset = set(stopwords.words('English'))
    x_check = [' '.join(w for w in sentence.split() if w.lower() not in stopset)
         for sentence in x
         ]
    #print(x_check)
    return x_check, y

def lemmatize(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word=word)

def sents_2(list_of_tweets):
    """
    Removing stop words
    ---
    Synsets english words into sets of synonyms
    -----
    Tokenization is breaking the sentence into words and punctuation.

    """
    stopwords = nltk.corpus.stopwords.words('english')
    contextTerms = []
    for sent in list_of_tweets:
        for word in sent.split():
            word_lemmatizer = WordNetLemmatizer()
            word = word_lemmatizer.lemmatize(word.lower())
            if wordnet.synsets(word) and word not in stopwords and len(word)>2:
                contextTerms.append(word)

    #print( contextTerms)
    return contextTerms

def tweets_words_sentiment(path,tweets):
    words_list = [line.rstrip('\n') for line in
                  open(path)]  # list of  sentiment words for looping
    dictSents = []
    words_of_combined_tweets = []
    words_of_combined_sentiment = []
    for word in words_list:
        aword, sentiment = word.split(",")
        tempList = []
        for tweet in tweets:
            if aword in tweet:
               # print("WITH TAGGING : " + word + " A Word : " + aword + ' CAUGHT IN ' + tweet + ' times word at ' + 'with sentiment ' + sentiment)
                tempList.append(tweet)
                #print(tempList)
        tweets_combined = ' '.join(tempList).strip()
        if tweets_combined:  # if tweet combined is not empty
            dictSents.append(tweets_combined)  # dict sent all
            words_of_combined_tweets.append(aword)  # single word
            words_of_combined_sentiment.append(sentiment)  # only sentiments

    # print('TWEETS WITH OCCURING WORDS ARE')
    # print(dictSents)
    # #print(len(dictSents))
    # print(' WORDS OF TWEETS ARE')
    # print(words_of_combined_tweets)
    # print('WORDS OF TWEETS WITH SENTIMENT')
    # print(words_of_combined_sentiment)

    #print(len(words_of_combined_sentiment))

    return dictSents, words_of_combined_tweets, words_of_combined_sentiment


def bagOfWordsWithList(sentenses):
    result = []
    for sent in sentenses:
        words = sent.split(' ')
        if len(words) > 3:
            result.append(words)
    return result


def bagOfWordsForWords(path):
    tweets =[]
    with open(path, "r") as textFile:
        reader = csv.reader(textFile, delimiter ='\t')
        for row in reader:
            tweets.append(row)
        tweets = [val for sublist in tweets for val in sublist]
       # print('Words ')
        #print(tweets)
        return tweets

def vectorizer(context_term, dictsent):
    vectorizer = TfidfVectorizer()
    fit_vector = vectorizer.fit(context_term) #learn the vocabulary of dict
    trns_vector = fit_vector.transform(dictsent)
   # print(trns_vector)
    cfd = pd.DataFrame(trns_vector.todense(), columns=vectorizer.get_feature_names())
    #print(cfd)
    return cfd

def k_fold(mat, y_label):
    kf = sklearn.model_selection.KFold(n_splits=2, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(mat):
        print("Train: ", train_index," Test :", test_index)
        x_train, x_test = mat[train_index], mat[test_index]
        y_train, y_test = y_label[train_index],y_label[test_index]

    return x_train,x_test, y_train, y_test

#data can undestand assingn the values of data to numbers

#Another approach to encoding categorical values is to use a technique called label encoding.
#Label encoding is simply converting each value in a column to a number.

def labelEncoding(y): 
    labelEncoder = LabelEncoder()
    y_encoded = labelEncoder.fit_transform(y)
    y_encoded
    return y_encoded

#remove stopwords this an the
#covert documents to a matrix of token count
# catagorical grouping  feacture extraction

# def countVectorizer(x):
#     stopset = set(stopwords.words('English'))
#     vect = CountVectorizer(analyzer='word', encoding='utf-8', min_df = 0, ngram_range=(1, 1), lowercase = True, strip_accents='ascii', stop_words = stopset)
#     X_vec = vect.fit_transform(x)
#     return X_vec

#convert  a raw document to a matrix of TD- IDF feature
#equivalent to countvectorizer
# take tweet  

def tfidfVectorizer(x):
    stopset = set(stopwords.words('English'))
    vect = TfidfVectorizer(analyzer='word', encoding='utf-8', min_df = 0, ngram_range=(1, 1), lowercase = True, strip_accents='ascii', stop_words = stopset)
    X_vec = vect.fit_transform(x)
    return X_vec

def tfidfVectorizer_new(x,dict_sent):
    stopset = dict_sent
    vect = TfidfVectorizer(analyzer='word', encoding='utf-8', min_df = 0, ngram_range=(1, 1), lowercase = True, strip_accents='ascii', stop_words = stopset)
    X_vec = vect.fit_transform(x)
    return X_vec

#split array into training and test data
def splitTestTrain(X_vec, y_encoded):
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y_encoded, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test
#by cross validataion
#we need to 
def kFold(X_vec, y_encoded):
    kf = KFold(n_splits=2, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(X_vec):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_vec[train_index], X_vec[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
        return X_train, X_test, y_train, y_test



def plotPreRec(naiveBayesRecall, naiveBayesPrecision, svmRecall, svmPrecision, randomForestRecall, randomForestPrecision):
    plt.plot([naiveBayesRecall],[naiveBayesPrecision], 'ro')
    plt.plot([svmRecall],[svmPrecision], 'ms')
    plt.plot([randomForestRecall],[randomForestPrecision], 'yo')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall comparison plot')
    plt.legend(['MNB', 'SVM', 'RF'], loc='upper left')
    plt.show() 
    
def plotAcuuracyComaprisonGraph(naiveBayesFMeasure, svmFMeasure, randomForestFMeasure):
    # Accuracy Comparison Plot
    cl = ('MNB', 'SVC', 'RF')
    y_pos = np.arange(len(cl))
    acc = [77.2682926829,79.0243902439,76.8780487805]
    plt.bar(y_pos, acc, align='center', alpha=0.5)
    plt.xticks(y_pos, cl)
    plt.title('Accuracy Comparison Plot')
    plt.show()
    cl = ('MNB', 'SVC', 'RF')
    y_pos = np.arange(len(cl))
    acc = [naiveBayesFMeasure, svmFMeasure, randomForestFMeasure]
    plt.bar(y_pos, acc, align='center', alpha=1.0)
    plt.xticks(y_pos, cl)
    plt.title('F Measure Comparison Plot')
    plt.show()

def applyMultinomialNBC(x_train, y_train, x_test, y_test):
    clf = MultinomialNB().fit(x_train, y_train)
    print(clf.score(x_test, y_test))
    y_score = clf.predict(x_test)
    precision = average_precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score, average='macro')
    f = 2 * (precision * recall) / (precision + recall)
    accuracy = metrics.accuracy_score(y_test, y_score)
    print("Multinomial Naive Bayes Classifier Test Accuracy: ", accuracy * 100)
    print("Multinomial Naive Bayes Classifier Test Precision: ", precision * 100)
    print("Multinomial Naive Bayes Classifier Test Recall: ", recall * 100)
    print("Multinomial Naive Bayes Classifier Test F measure: ", f * 100)

    return precision, recall, f

def applySVMClassifier(x_train, y_train, x_test, y_test):
    # Model Training: SVMs
    svc_classifier = SVC(kernel='linear', random_state=0)
    svc_classifier.fit(x_train, y_train)
    model_accuracies = cross_val_score(estimator=svc_classifier, X=x_train, y=y_train, cv=10)
    print("Model Accuracies Mean", model_accuracies.mean()*100)
    print("Model Accuracies Standard Devision", model_accuracies.std()*100)
    # Model Testing: SVMs
    y_pred = svc_classifier.predict(x_test)
    metrics.confusion_matrix(y_test, y_pred)
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    precision_SVC = precision_score(y_test, y_pred, average='macro')  
    recall_SVC = recall_score(y_test, y_pred, average='macro') 
    f_SVC = 2*(precision_SVC * recall_SVC) / (precision_SVC + recall_SVC)
    print("SVCs Test Accuracy: ", test_accuracy*100)
    print("SVCs Test Precision: ", precision_SVC*100)
    print("SVCs Test Recall: ", recall_SVC*100)
    print("SVCs Test F measure: ", f_SVC*100)
    return precision_SVC, recall_SVC, f_SVC
    
def applyRandomForestClassifier(x_train, y_train, x_test, y_test):
    # Model Training: Random Forests Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, class_weight="balanced", criterion='entropy', random_state=1)
    rf_classifier.fit(x_train, y_train)
    model_accuracies = cross_val_score(estimator=rf_classifier, X=x_train, y=y_train, cv=5)
    print("Model Accuracies Mean", model_accuracies.mean()*100)
    print("Model Accuracies Standard Devision", model_accuracies.std()*100)
    # Model Testing: Random Forests Classifier
    y_pred = rf_classifier.predict(x_test)
    metrics.confusion_matrix(y_test, y_pred)
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f = 2*(pre * recall) / (pre + recall)
    print("Random Forests Test Accuracy: ", test_accuracy*100)
    print("Random Forests Test Precision: ", pre*100)
    print("Random Forests Test Recall: ", recall*100)
    print("Random Forests Test F measure: ", f*100)
    return pre, recall, f
    


def plotLabels(y):
    #Encoding y
    y_encoded = labelEncoding(y)
    #Count Labels and plot them
    y_count = Counter(y_encoded)
    key = y_count.keys()
    df = pd.DataFrame(y_count,index=key)
    df.drop(df.columns[1:], inplace=True)
    df.plot(kind='bar')
    plt.show()
