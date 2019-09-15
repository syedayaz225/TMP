import twitter_senti as ts

"""
"""
    #Testing Data
filePathWords = "data//Posive and negative words new.txt"
filePathCorpus = "data//short Dataset.csv" #
file_path_standard_words = "data//Posive and negative words new with tagging comma.txt"

#     #Original data
# filePathWords = "data//Posive and negative words new.txt"
# filePathCorpus = "data//data.csv"
# file_path_standard_words = "data//Posive and negative words new with tagging.txt"


#cleaning data
sent = ts.sents(filePathCorpus)
#removing stopwords and tokanizing
context_term= ts.sents_2(sent[0])
#list of tweets
tweets = sent[0]


"""
#comparision between tweets and standard word

Parameters
Input
:file_path_standard_words : 
:tweets : 
Output
:dictSents
:words_of_combined_words
:words_of_combined_sentiment : 
"""
dictSents, words_of_combined_words, words_of_combined_sentiment = ts.tweets_words_sentiment(file_path_standard_words,tweets)

y_label = words_of_combined_sentiment
#TFID Vectorizer
#Convert a collection of raw documents(dicSent) to a matrix of TF-IDF features.
x_vec = ts.vectorizer(context_term,dictSents)


y_encoded = ts.labelEncoding(y_label)
#print("Encoded")
#print(y_encoded)
#ts.plotLabels(y_encoded)
#X_vec = ts.tfidfVectorizer_new(x_vec,y_encoded)
#x_train,x_test , y_train, y_test = ts.splitTestTrain(x_vec, y_encoded)


x_train, x_test, y_train, y_test = ts.train_test_split(x_vec, y_encoded, test_size=0.33, random_state=42)

#x_train,x_test , y_train, y_test = ts.kFold(x_vec, y_encoded)
naiveBayesPrecision, naiveBayesRecall, naiveBayesFMeasure = ts.applyMultinomialNBC(x_train, y_train, x_test, y_test)
randomForestPrecision, randomForestRecall, randomForestFMeasure = ts.applyRandomForestClassifier(x_train, y_train, x_test, y_test)
svmPrecision, svmRecall, svmFMeasure = ts.applySVMClassifier(x_train,y_train,x_test, y_test)

# TODO: Add more graphy
#Plot Precision-Recall comparison graph
ts.plotPreRec(naiveBayesRecall, naiveBayesPrecision, svmRecall, svmPrecision, randomForestRecall, randomForestPrecision)
#plot FMeasure comparison graph
ts.plotAcuuracyComaprisonGraph(naiveBayesFMeasure, svmFMeasure, randomForestFMeasure )

