Usage
-----

	\program <path_for_train_spam> <path_for_train_ham> <path_for_test_spam> <path_for_test_ham> <path_for_stopwords> <learning_rate> <number_of_loops_for_convergence>

Usage example
-------------

	py hw3.py ./train/spam/ ./train/ham/ ./test/spam/ ./test/ham/ ./stopwords.txt 0.2 50

PLease use attached stopwords.txt.

My testing results
------------------

	Input: py hw3.py ./train/spam/ ./train/ham/ ./test/spam/ ./test/ham/ ./stopwords.txt 0.2 10

	Output:
	***************************OUTPUT*****************************
	Accuracy for Perceptron with stopwords: 0.6380753138075314
	Accuracy for Perceptron without stopwords: 0.9456066945606695

Training and testing dataset
----------------------------

You can find training and testing dataset in "Naive Bayes and Logistic Regression" folder in current repository.