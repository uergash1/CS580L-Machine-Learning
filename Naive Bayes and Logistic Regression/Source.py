from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import random
import numpy as np
import os, os.path
import re
import string
import glob 
import codecs
import math

class dataNB:
    def __init__(self):
        self.train_spam = []
        self.train_ham = []
        self.test_spam_path = ""
        self.test_ham_path = ""
        self.test_spam_number = 0
        self.test_ham_number = 0
        self.train_spam_number = 0
        self.train_ham_number = 0
        self.stopwords_path = ""

class dataLR:
    def __init__(self):
        self.weight = []
        self.vocabulary = []
        self._lambda = 0
        self.learning_rate = 0
        self.convergence_loop = 0
        self.stopwords_path = ""

#remove special characters
def remove_special_characters(source):
    return re.sub(r"[,"":.;@#?!&$'`~^*]+\ *", " ", source)

#removing punctuations
def remove_punctuation(source):
    return re.sub('['+string.punctuation+']', ' ', source)
    
#remove single letter from a string
def remove_single_letter(source):
    return ' '.join( [w for w in source.split() if len(w)>1] )

#tokenization and removing stop words
def tokenization(source):
    tokens = word_tokenize(source)
    tokens = [token.lower() for token in tokens]
    return tokens

#tokenization and removing stop words
def remove_stopwords(source, stpath):
    stoptxt = open(stpath, 'r', encoding='utf-8')
    stopwords = stoptxt.read()
    stoptxt.close()
    stopwords = tokenization(stopwords)
    filteredTokens = []
    for token in source:
        if token not in stopwords:
            filteredTokens.append(token)
    return filteredTokens

#Porter's stemming
def porter_stemming(source):
    ps = PorterStemmer()
    stemmedTokens = []
    for token in source:
        stemmedTokens.append(ps.stem(token))
    return stemmedTokens

def get_all_dataset(path, stpath):
    full_path = path + '*.txt'    
    files  = glob.glob(full_path)
    tokens = []
    for file in files:
        with codecs.open(file, "r",encoding='utf-8', errors='ignore') as fdata:
            source = fdata.read()
        source = remove_special_characters(source)
        source = remove_punctuation(source)
        source = remove_single_letter(source)
        source = tokenization(source)

        if stpath != "":
            source = remove_stopwords(source, stpath)

        source = porter_stemming(source)
        tokens = tokens + source
    return tokens

def get_dataset(path, stpath):
    with codecs.open(path, "r",encoding='utf-8', errors='ignore') as fdata:
        source = fdata.read()
    source = remove_special_characters(source)
    source = remove_punctuation(source)
    source = remove_single_letter(source)
    source = tokenization(source)
    if stpath != "":
        source = remove_stopwords(source, stpath)
    source = porter_stemming(source)
    return source

def get_number_files(path):
    return len([f for f in os.listdir(path)])

def naive_bayes(dataset):
    #****************************************************
    # predicting test spam
    #****************************************************
    full_path = dataset.test_spam_path + '*.txt'    
    test_spam_files  = glob.glob(full_path)
    
    num_dist_tok_train = len(set(dataset.train_spam)) + len(set(dataset.train_ham))
    
    prior_spam = dataset.train_spam_number / (dataset.train_spam_number + dataset.train_ham_number)
    prior_spam = math.log(prior_spam, 2)

    prior_ham = dataset.train_ham_number / (dataset.train_spam_number + dataset.train_ham_number)
    prior_ham = math.log(prior_ham, 2)

    #counting number of occurences of the token in train spam
    correct_predict_spam = 0
    for test_spam_file in test_spam_files:
        
        total_likelihood_spam = 0
        total_likelihood_ham = 0
        test_spam_dataset = get_dataset(test_spam_file, dataset.stopwords_path)

        for test_token in test_spam_dataset:
            
            number_occur_train_spam = 0
            number_occur_train_ham = 0

            number_occur_train_spam = dataset.train_spam.count(test_token)
            number_occur_train_ham = dataset.train_ham.count(test_token)

            likelihood_spam = (number_occur_train_spam + 1) / (len(dataset.train_spam) + num_dist_tok_train)
            likelihood_spam = math.log(likelihood_spam, 2)
            total_likelihood_spam = total_likelihood_spam + likelihood_spam

            likelihood_ham = (number_occur_train_ham + 1) / (len(dataset.train_ham) + num_dist_tok_train)
            likelihood_ham = math.log(likelihood_ham, 2)
            total_likelihood_ham = total_likelihood_ham + likelihood_ham

        postirior_spam = prior_spam + total_likelihood_spam
        postirior_ham = prior_ham + total_likelihood_ham

        if(postirior_spam >= postirior_ham):
            correct_predict_spam = correct_predict_spam + 1

    #****************************************************
    # predicting test ham
    #****************************************************
    full_path = dataset.test_ham_path + '*.txt'    
    test_ham_files  = glob.glob(full_path)

    #counting number of occurences of the token in train spam
    correct_predict_ham = 0
    for test_ham_file in test_ham_files:
        
        total_likelihood_spam = 0
        total_likelihood_ham = 0
        test_ham_dataset = get_dataset(test_ham_file, dataset.stopwords_path)

        for test_token in test_ham_dataset:
            
            number_occur_train_spam = 0
            number_occur_train_ham = 0

            number_occur_train_spam = dataset.train_spam.count(test_token)
            number_occur_train_ham = dataset.train_ham.count(test_token)

            likelihood_spam = (number_occur_train_spam + 1) / (len(dataset.train_spam) + num_dist_tok_train)
            likelihood_spam = math.log(likelihood_spam, 2)
            total_likelihood_spam = total_likelihood_spam + likelihood_spam

            likelihood_ham = (number_occur_train_ham + 1) / (len(dataset.train_ham) + num_dist_tok_train)
            likelihood_ham = math.log(likelihood_ham, 2)
            total_likelihood_ham = total_likelihood_ham + likelihood_ham

        postirior_spam = prior_spam + total_likelihood_spam
        postirior_ham = prior_ham + total_likelihood_ham

        if(postirior_spam < postirior_ham):
            correct_predict_ham = correct_predict_ham + 1

    number_test_files = dataset.test_spam_number + dataset.test_ham_number
    return {"correct_predict_ham": correct_predict_ham, "correct_predict_spam": correct_predict_spam, "number_test_files": number_test_files}

def train_LR(train_path, dataset):
    train_spam_examples = []
    train_ham_examples = []
    data = []
    full_path = train_path['spam'] + '*.txt'    
    train_spam_paths  = glob.glob(full_path)
    for train_spam_path in train_spam_paths:
        train_spam_example = get_dataset(train_spam_path, dataset.stopwords_path)
        train_spam_examples.append(train_spam_example)

    full_path = train_path['ham'] + '*.txt'    
    train_ham_paths  = glob.glob(full_path)
    for train_ham_path in train_ham_paths:
        train_ham_example = get_dataset(train_ham_path, dataset.stopwords_path)
        train_ham_examples.append(train_ham_example)

    t1 = [y for x in train_spam_examples for y in x]
    t2 = [y for x in train_ham_examples for y in x]

    dataset.vocabulary = list(set(t1 + t2)) 

    number_train_docs = get_number_files(train_path['spam']) + get_number_files(train_path['ham'])
    vocabulary_size = len(dataset.vocabulary)

    row = 0
    for example in train_spam_examples:
        data.append([])
        data[row] = [0]*(vocabulary_size + 2)
        data[row][0] = 1
        tokens = dict(Counter(example))
        for key in tokens.keys():
            position = dataset.vocabulary.index(key)
            data[row][position + 1] = tokens[key]
        data[row][vocabulary_size + 1] = 1
        row = row + 1

    for example in train_ham_examples:
        data.append([])
        data[row] = [0]*(vocabulary_size + 2)
        data[row][0] = 1
        tokens = dict(Counter(example))
        for key in tokens.keys():
            position = dataset.vocabulary.index(key)
            data[row][position + 1] = tokens[key]
        data[row][vocabulary_size + 1] = 0
        row = row + 1

    Pr = np.random.uniform(low=0.0, high=1.0, size=(number_train_docs))
    dataset.weight = np.random.uniform(low=0.0, high=1.0, size=(vocabulary_size + 1))
    
    for convergence in range(int(dataset.convergence_loop)):

        #calculating Pr fo each example
        for row in range(number_train_docs):

            #calculating power of exponent
            power = 0
            for k in range(vocabulary_size + 1):
                power = power + (dataset.weight[k] * data[row][k])

            try:
                Pr[row] = math.exp(power) / (1 + math.exp(power))
            except OverflowError:
                Pr[row] = 0.999

        dw = []
        dw = [0]*(vocabulary_size + 1)
        dw = np.array(dw)

        data = np.array(data)
        temp = data[:,:-1].transpose()

        temp1 = temp.dot(data[:,vocabulary_size + 1] - Pr)
        dw = dw + temp1
        dataset.weight = dataset.weight + (float(dataset.learning_rate) * (dw - (float(dataset._lambda) * dataset.weight)))

def test_LG(test_path, dataset):
    correct_predict_spam = 0
    correct_predict_ham = 0
    full_path = test_path['spam'] + '*.txt'    
    test_spam_paths  = glob.glob(full_path)
    for test_spam_path in test_spam_paths:
        test_spam_example = get_dataset(test_spam_path, dataset.stopwords_path)
        test_spam_tokens = dict(Counter(test_spam_example))
        power = dataset.weight[0]

        for token in test_spam_tokens.keys():
            if token in dataset.vocabulary:
                position = dataset.vocabulary.index(token)
                power = power + (test_spam_tokens[token] * dataset.weight[position + 1])
        try:
            p = math.exp(power) / (1 + math.exp(power))
        except OverflowError:
            p = 0.999
        
        if(p >= 0.5):
            correct_predict_spam = correct_predict_spam + 1

    full_path = test_path['ham'] + '*.txt'    
    test_ham_paths  = glob.glob(full_path)
    for test_ham_path in test_ham_paths:
        test_ham_example = get_dataset(test_ham_path, dataset.stopwords_path)
        test_ham_tokens = dict(Counter(test_ham_example))
        power = dataset.weight[0]

        for token in test_ham_tokens.keys():
            if token in dataset.vocabulary:
                position = dataset.vocabulary.index(token)
                power = power + (test_ham_tokens[token] * dataset.weight[position + 1])
        try:
            p = math.exp(power) / (1 + math.exp(power))
        except OverflowError:
            p = 0.999

        if(p < 0.5):
            correct_predict_ham = correct_predict_ham + 1

    number_test_files = get_number_files(test_path['spam']) + get_number_files(test_path['ham'])

    return {'correct_predict_spam': correct_predict_spam, 'correct_predict_ham': correct_predict_ham, 'number_test_files': number_test_files}

#random feature selection for Logistic Regression
def random_feature_selection(dataset):
    vocabulary_size = len(dataset.vocabulary)
    random_feature_size = random.randint(int(vocabulary_size * 0.75), int(vocabulary_size))
    remove_features = vocabulary_size - random_feature_size
    for i in range(remove_features):
        index = random.randint(1, vocabulary_size-1)
        token = dataset.vocabulary[index]
        dataset.vocabulary.pop(index)
        dataset.weight = np.delete(dataset.weight, index)
        vocabulary_size = vocabulary_size - 1

def get_accuracy(result):
    accuracy = (result['correct_predict_ham'] + result['correct_predict_spam']) / result['number_test_files']
    return accuracy

def main():

    ########################################################################
    # User input variables
    ########################################################################

    train_spam_path = input("Enter the path for train spam files: ")
    assert os.path.exists(train_spam_path), "There is no such a directory "+str(train_spam_path)

    train_ham_path = input("Enter the path for train ham files: ")
    assert os.path.exists(train_ham_path), "There is no such a directory "+str(train_ham_path)

    test_spam_path = input("Enter the path for test spam files: ")
    assert os.path.exists(test_spam_path), "There is no such a directory "+str(test_spam_path)

    test_ham_path = input("Enter the path for test ham files: ")
    assert os.path.exists(test_ham_path), "There is no such a directory "+str(test_ham_path)

    stpath = input("Enter the path for stop words txt file: ")
    assert os.path.exists(stpath), "There is no such a file "+str(stpath)

    learning_rate = input("Enter learning rate for LR: ")
    lamda = input("Enter lambda for LR: ")
    convergence_loop = input("Enter number of iterations for regularization convergence: ")
    
    print("***************************OUTPUT*****************************")    

    ########################################################################
    # Naive Bayes with stop words
    ########################################################################

    datasetNB = dataNB()

    datasetNB.train_spam = get_all_dataset(train_spam_path, "")
    datasetNB.train_spam_number = get_number_files(train_spam_path)

    datasetNB.train_ham = get_all_dataset(train_ham_path, "")
    datasetNB.train_ham_number = get_number_files(train_ham_path)

    datasetNB.test_spam_path = test_spam_path
    datasetNB.test_spam_number = get_number_files(test_spam_path)

    datasetNB.test_ham_path = test_ham_path
    datasetNB.test_ham_number = get_number_files(test_ham_path)

    result = naive_bayes(datasetNB)

    print("Accuracy for NB with stopwords:",get_accuracy(result))

    ########################################################################
    # Naive Bayes without stop words
    ########################################################################

    datasetNB_stwords = dataNB()
    
    datasetNB_stwords.stopwords_path = stpath

    datasetNB_stwords.train_spam = get_all_dataset(train_spam_path, stpath)
    datasetNB_stwords.train_spam_number = get_number_files(train_spam_path)

    datasetNB_stwords.train_ham = get_all_dataset(train_ham_path, stpath)
    datasetNB_stwords.train_ham_number = get_number_files(train_ham_path)

    datasetNB_stwords.test_spam_path = test_spam_path
    datasetNB_stwords.test_spam_number = get_number_files(test_spam_path)

    datasetNB_stwords.test_ham_path = test_ham_path
    datasetNB_stwords.test_ham_number = get_number_files(test_ham_path)

    result = naive_bayes(datasetNB_stwords)

    print("Accuracy for NB without stopwords:",get_accuracy(result))

    ##########################################################
    # Logistic Regression with stop words
    ##########################################################

    datasetLR = dataLR()

    datasetLR.learning_rate = learning_rate
    datasetLR._lambda = lamda
    datasetLR.convergence_loop = convergence_loop

    train_path = {'spam': train_spam_path, 'ham': train_ham_path}
    test_path = {'spam': test_spam_path, 'ham': test_ham_path}

    train_LR(train_path, datasetLR)
    result = test_LG(test_path, datasetLR)

    print("Accuracy for LR with stopwords:",get_accuracy(result))

    ##########################################################
    # Logistic Regression without stop words
    ##########################################################

    datasetLR_stopwords = dataLR()

    datasetLR_stopwords.learning_rate = learning_rate
    datasetLR_stopwords._lambda = lamda
    datasetLR_stopwords.convergence_loop = convergence_loop
    datasetLR_stopwords.stopwords_path = stpath

    train_LR(train_path, datasetLR_stopwords)
    result = test_LG(test_path, datasetLR_stopwords)

    print("Accuracy for LR without stopwords:",get_accuracy(result))
    
    ##########################################################
    # Feature Selection for Logistic Regression
    ##########################################################

    random_feature_selection(datasetLR)
    train_LR(train_path, datasetLR)
    result = test_LG(test_path, datasetLR)
    print("Accuracy for Feature Selection:",get_accuracy(result))

if __name__ == "__main__":
    main()