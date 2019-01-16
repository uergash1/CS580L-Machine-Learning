from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import numpy as np
import os, os.path
import re
import string
import glob
import codecs
import sys
import ast


class data:
    def __init__(self):
        self.weight = []
        self.vocabulary = []
        self._lambda = 0
        self.learning_rate = 0.0
        self.convergence_loop = 0
        self.stopwords_path = ""

# remove numbers
def remove_numbers(source):
    return re.sub(r'\d+', '', source)

# remove special characters
def remove_special_characters(source):
    return re.sub(r"[,"":.;@#?!&$'`~^*]+\ *", " ", source)

# removing punctuations
def remove_punctuation(source):
    return re.sub('[' + string.punctuation + ']', ' ', source)

# remove single letter from a string
def remove_single_letter(source):
    return ' '.join([w for w in source.split() if len(w) > 1])

# tokenization and removing stop words
def tokenization(source):
    tokens = word_tokenize(source)
    tokens = [token.lower() for token in tokens]
    return tokens

# tokenization and removing stop words
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

# Porter's stemming
def porter_stemming(source):
    ps = PorterStemmer()
    stemmedTokens = []
    for token in source:
        stemmedTokens.append(ps.stem(token))
    return stemmedTokens

def get_all_dataset(files, stpath):
    # full_path = path + '*.txt'
    # files = glob.glob(full_path)
    tokens = []
    for file in files:
        with codecs.open(file, "r", encoding='utf-8', errors='ignore') as fdata:
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
    with codecs.open(path, "r", encoding='utf-8', errors='ignore') as fdata:
        source = fdata.read()
    source = remove_numbers(source)
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

# fill data matrix with spam and ham inputs
def get_data_grid(train_path, dataset):
    train_spam_examples = []
    train_ham_examples = []
    data = []
    full_path = train_path['spam'] + '*.txt'
    train_spam_paths = glob.glob(full_path)
    for train_spam_path in train_spam_paths:
        train_spam_example = get_dataset(train_spam_path, dataset.stopwords_path)
        train_spam_examples.append(train_spam_example)

    full_path = train_path['ham'] + '*.txt'
    train_ham_paths = glob.glob(full_path)
    for train_ham_path in train_ham_paths:
        train_ham_example = get_dataset(train_ham_path, dataset.stopwords_path)
        train_ham_examples.append(train_ham_example)

    t1 = [y for x in train_spam_examples for y in x]
    t2 = [y for x in train_ham_examples for y in x]

    dataset.vocabulary = list(set(t1 + t2))

    #number_train_docs = get_number_files(train_path['spam']) + get_number_files(train_path['ham'])
    vocabulary_size = len(dataset.vocabulary)

    row = 0
    for example in train_spam_examples:
        data.append([])
        data[row] = [0] * (vocabulary_size + 2)
        data[row][0] = 1
        tokens = dict(Counter(example))
        for key in tokens.keys():
            position = dataset.vocabulary.index(key)
            data[row][position + 1] = tokens[key]
        data[row][vocabulary_size + 1] = 1
        row = row + 1

    for example in train_ham_examples:
        data.append([])
        data[row] = [0] * (vocabulary_size + 2)
        data[row][0] = 1
        tokens = dict(Counter(example))
        for key in tokens.keys():
            position = dataset.vocabulary.index(key)
            data[row][position + 1] = tokens[key]
        data[row][vocabulary_size + 1] = -1
        row = row + 1

    # print(len(data))
    # print(len(data[0]))
    return data

# def create_csv(data, dataset):
#     with open('csvfile.csv', 'w') as file:
#         str2 = "bais,"
#         for token in dataset.vocabulary:
#             str2 = str2 + token + ','
#         file.write(str2)
#         file.write('\n')

#         for example in data:
#             str1 = ""
#             for tf in example:
#                 str1 = str1 + str(tf) + ','
#             file.write(str1)
#             file.write('\n')


def predict(example, weight):
    sum = 0.0
    for i in range(len(weight)):
        sum = sum + example[i] * weight[i]
    if sum > 0:
        return 1
    return -1

def train_perceptron(train_path, dataset):
    data = get_data_grid(train_path, dataset)
    data = np.array(data)
    vocabulary_size = len(dataset.vocabulary)
    number_train_docs = len(data)

    # output generated by the perceptron
    o = np.random.uniform(low=0, high=0, size=number_train_docs)

    # target output for all training examples
    t = data[:, vocabulary_size + 1]

    temp = data[:, :-1].transpose()

    # filling weight with random values
    dataset.weight = np.random.uniform(low=-1.0, high=1.0, size=(vocabulary_size + 1))

    for convergence in range(int(dataset.convergence_loop)):
        # calculating output for each example
        for row in range(number_train_docs):
            o[row] = predict(data[row], dataset.weight)

        delta_weight = dataset.learning_rate * temp.dot(t - o)
        dataset.weight = dataset.weight + delta_weight

# test perceptron
def test_perceptron(test_path, dataset):
    correct_predict_spam = 0
    correct_predict_ham = 0
    full_path = test_path['spam'] + '*.txt'
    test_spam_paths = glob.glob(full_path)
    for test_spam_path in test_spam_paths:
        test_spam_example = get_dataset(test_spam_path, dataset.stopwords_path)
        test_spam_tokens = dict(Counter(test_spam_example))
        sum = dataset.weight[0]

        for token in test_spam_tokens.keys():
            if token in dataset.vocabulary:
                position = dataset.vocabulary.index(token)
                sum = sum + (test_spam_tokens[token] * dataset.weight[position + 1])
        if sum > 0:
            correct_predict_spam = correct_predict_spam + 1

    full_path = test_path['ham'] + '*.txt'
    test_ham_paths = glob.glob(full_path)
    for test_ham_path in test_ham_paths:
        test_ham_example = get_dataset(test_ham_path, dataset.stopwords_path)
        test_ham_tokens = dict(Counter(test_ham_example))
        sum = dataset.weight[0]

        for token in test_ham_tokens.keys():
            if token in dataset.vocabulary:
                position = dataset.vocabulary.index(token)
                sum = sum + (test_ham_tokens[token] * dataset.weight[position + 1])
        if sum <= 0:
            correct_predict_ham = correct_predict_ham + 1

    number_test_files = get_number_files(test_path['spam']) + get_number_files(test_path['ham'])

    return {'correct_predict_spam': correct_predict_spam, 'correct_predict_ham': correct_predict_ham,
            'number_test_files': number_test_files}

def get_accuracy(result):
    accuracy = (result['correct_predict_ham'] + result['correct_predict_spam']) / result['number_test_files']
    return accuracy

def main():

    command_line_args = str(sys.argv)
    command_line_args = ast.literal_eval(command_line_args)
    print(len(command_line_args))

    if len(command_line_args) != 8:
        print("Invalid command line parameters!")
        print(
            "Usage: .\program <path_for_train_spam> <path_for_train_ham> <path_for_test_spam> "
            "<path_for_test_ham> <path_for_stopwords> "
            "<learning_rate> <number_of_loops_for_convergence>")
    else:
        train_spam = str(command_line_args[1])
        train_ham = str(command_line_args[2])
        test_spam = str(command_line_args[3])
        test_ham = str(command_line_args[4])
        stopwords_path = str(command_line_args[5])
        learning_rate = str(command_line_args[6])
        convergence_loop = str(command_line_args[7])
        
        print("***************************OUTPUT*****************************")
        
		##########################################################
	    # Perceptron with stop words
	    ##########################################################

        dataset = data()
        dataset.learning_rate = float(learning_rate)
        dataset.convergence_loop = int(convergence_loop)
        train_path = {'spam': train_spam, 'ham': train_ham}
        test_path = {'spam': test_spam, 'ham': test_ham}
        train_perceptron(train_path, dataset)
        result = test_perceptron(test_path, dataset)
        print("Accuracy for Perceptron with stopwords:", get_accuracy(result))

        ##########################################################
        # Perceptron without stop words
        ##########################################################

        dataset_stopwords = data()
        dataset_stopwords.learning_rate = float(learning_rate)
        dataset_stopwords.convergence_loop = int(convergence_loop)
        dataset_stopwords.stopwords_path = "stopwords.txt"
        train_perceptron(train_path, dataset_stopwords)
        result = test_perceptron(test_path, dataset_stopwords)
        print("Accuracy for Perceptron without stopwords:", get_accuracy(result))

if __name__ == "__main__":
    main()