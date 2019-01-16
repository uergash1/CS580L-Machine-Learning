Problem statement
=================

Problem statement for this code you will find in Requirements.pdf

Input
-----

Program asks followings:

1. Path for train spam files (Example: ./train/spam/)
2. Path for train ham files (Example: ./train/ham/)
3. Path for test spam files (Example: ./test/spam/)
4. Path for test ham files (Example: ./test/ham/)
5. Path for stop words txt file (stopwords.txt document is attached)
6. Learning rate for Logistic Regression
7. Lambda for Logistic Regression
8. Number of iterations for regularization convergence

Output
------

Programs displays following outputs:

1. Accuracy for Naive Bayes with stopwords
2. Accuracy for Naive Bayes without stopwords
3. Accuracy for Logistic Regression with stopwords
4. Accuracy for Logistic Regression without stopwords
5. Accuracy for Feature Selection

Description
-----------

- If the accuracy for Naive Bayes with stopwords is high, then the reason for that can be the way how dataset is processed.

- Accuracy for Naive Bayes without stopwords did not increase for the given dataset. 
	
- Initial values of Pr and weight assigned randomly, that is why each time when the program is run, the accuracy for Logistic Regression is different.

- Accuracy for Logistic Regression without stopwords really depends on initial values of Pr and weight. If those initial values are proper, then accuracy for Logistic Regression without stopwords should increase.

- Feature Selection is implemented in the following way. Number for features' subset is picked randomly (For example: if the number of features is 8000 then, random_feature_size=random(8000*0.75, 8000)). Then, based on chosen feature size, features are also picked randomly. If chosen features are good, then the accuracy for Logistic Regression with stopwords increases.

- It takes time to process each algorithm, so please be patient!

- If you want to perform the code quicker for Logistic Regression, then input for regularization convergence should be smaller.

My one of the local outputs
---------------------------

	Enter the path for train spam files: ./train/spam/
	Enter the path for train ham files: ./train/ham/
	Enter the path for test spam files: ./test/spam/
	Enter the path for test ham files: ./test/ham/
	Enter the path for stop words txt file: ./stopwords.txt
	Enter learning rate for LR: 0.5
	Enter lambda for LR: 0.5
	Enter number of iterations for regularization convergence: 3

	Accuracy for NB with stopwords: 0.9623430962343096
	Accuracy for NB without stopwords: 0.9602510460251046
	Accuracy for LR with stopwords: 0.8556485355648535
	Accuracy for LR without stopwords: 0.9163179916317992
	Accuracy for Feature Selection: 0.8577405857740585