import nltk
import twitter_senti as ts
filePathWords = "data/Posive and negative words new.txt"
filePathCorpus = "data/dataTab.txt"
filePath20Tweets = "data/dataTab20.txt"


#ts.freqDist()

sent = ts.sents(filePath20Tweets)# cleaning data
print(sent)

#bg_sents = ts.bagOfWordsWithList(sent)
#frequent = nltk.FreqDist(sent)
#print(frequent)
#most_frequent= frequent.most_common(50)
#print(most_frequent)
#st_words = ts.bagOfWordsForWords(filePathWords) # standard words 


def applySGDClassifier(X_train, y_train, X_test, y_test):
    # Apply SGD Classifier
    SGDClassifier_classifier = SGDClassifier()
    SGDClassifier_classifier.fit(X_train, y_train)
    y_pred = SGDClassifier_classifier.predict(X_test)
    metrics.confusion_matrix(y_test, y_pred)
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    precision_SGD = precision_score(y_test, y_pred, average='macro')
    recall_SGD = recall_score(y_test, y_pred, average='macro')
    f_SGD = 2 * (precision_SGD * recall_SGD) / (precision_SGD + recall_SGD)
    print("SGD_classifier Accuracy percent:", test_accuracy * 100)
    print("SGD_classifier Precision percent:", precision_SGD * 100)
    print("SGD_classifier Recall percent:", recall_SGD * 100)
    print("SGD_classifier Recall F measure:", f_SGD * 100)
    return precision_SGD, recall_SGD, f_SGD

def applyLogisticRegressionClassifier(x_train, y_train, x_test, y_test):
    #Apply Logistic Regression Classifier
    lr = LogisticRegression(penalty = 'l2', C = 1)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    metrics.confusion_matrix(y_test, y_pred)
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f = 2*(pre * recall) / (pre + recall)
    print("LogisticRegression_classifier Accuracy percent:",test_accuracy *100)
    print("LogisticRegression_classifier Precision percent:",pre *100)
    print("LogisticRegression_classifier Recall percent:",recall *100)
    print("LogisticRegression_classifier F measure:",f*100)
    return pre, recall, f


def applyDecisionTreeClassifier(X_train, y_train, X_test, y_test):
    # Apply Decision Tree Classifier
    Decision_Tree_CLF = DecisionTreeClassifier(random_state=0)
    Decision_Tree_CLF.fit(X_train, y_train)
    y_pred = Decision_Tree_CLF.predict(X_test)
    metrics.confusion_matrix(y_test, y_pred)
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    precision_DT = precision_score(y_test, y_pred, average='macro')
    recall_DT = recall_score(y_test, y_pred, average='macro')
    f_DT = 2 * (precision_DT * recall_DT) / (precision_DT + recall_DT)
    print("SGD_classifier Accuracy percent:", test_accuracy * 100)
    print("SGD_classifier Precision percent:", precision_DT * 100)
    print("SGD_classifier Recall percent:", recall_DT * 100)
    print("SGD_classifier Recall F measure:", f_DT * 100)
    return precision_DT, recall_DT, f_DT

#ts.testNew(st_words,most_frequent,sent)



