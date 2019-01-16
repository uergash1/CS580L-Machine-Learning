import matplotlib.pyplot as plt
import naivebayes
import svm
import decisiontree
import logistic
import neuralnet
import twitter_driver
import utils
import preprocess
import stats

print('Processing training and testing dataset')
TRAIN_DATASET = "data/train.csv"
TEST_DATASET = "data/test.csv"
TRAIN_PROCESSED_DATASET = preprocess.run(TRAIN_DATASET, test_file=False)
TEST_PROCESSED_DATASET = preprocess.run(TEST_DATASET, test_file=True)
FREQ_DIST_FILE, BI_FREQ_DIST_FILE = stats.run(TRAIN_PROCESSED_DATASET)


results = {}

accSVM = svm.run(FREQ_DIST_FILE, BI_FREQ_DIST_FILE, TRAIN_PROCESSED_DATASET, TEST_PROCESSED_DATASET, False)
results['SVM'] = accSVM


accNB = naivebayes.run(FREQ_DIST_FILE, BI_FREQ_DIST_FILE, TRAIN_PROCESSED_DATASET, TEST_PROCESSED_DATASET, False)
results['Naive Bayes'] = accNB


accDT = decisiontree.run(FREQ_DIST_FILE, BI_FREQ_DIST_FILE, TRAIN_PROCESSED_DATASET, TEST_PROCESSED_DATASET, False)
results['Decision Tree'] = accDT


accL = logistic.run(FREQ_DIST_FILE, BI_FREQ_DIST_FILE, TRAIN_PROCESSED_DATASET, TEST_PROCESSED_DATASET, False)
results['Logistic Regression'] = accL


accNN = neuralnet.run(FREQ_DIST_FILE, BI_FREQ_DIST_FILE, TRAIN_PROCESSED_DATASET, TEST_PROCESSED_DATASET, False)
results['Neural Network'] = accNN
print("\naccNB:", accNB, "accSVM:", accSVM, "accDT:", accDT, "accL:", accL, "accNN:", accNN)

plt.bar(range(len(results)), results.values(), align='center')
plt.xticks(range(len(results)), list(results.keys()), rotation='vertical')
plt.show()

searchTerm = input("\nEnter keyword/hashtag to search about: ")
noOfSearchTerms = int(input("Enter how many tweets to analyze: "))

print('Downloading real-time data from Twitter and processing it')
tweets = twitter_driver.get_tweets(searchTerm, noOfSearchTerms)
tweets_csv = utils.save_tweets(tweets)
preprocess.run(tweets_csv, test_file=True)

processed_tweets_csv = tweets_csv[:-4] + "-processed.csv"
TEST_PROCESSED_DATASET = processed_tweets_csv

max_accuracy = max(list(results.values()))
if max_accuracy == results['Neural Network']:
    predictions = neuralnet.run(FREQ_DIST_FILE, BI_FREQ_DIST_FILE, TRAIN_PROCESSED_DATASET, TEST_PROCESSED_DATASET, True)
elif max_accuracy == results['Decision Tree']:
    predictions = decisiontree.run(FREQ_DIST_FILE, BI_FREQ_DIST_FILE, TRAIN_PROCESSED_DATASET, TEST_PROCESSED_DATASET, True)
elif max_accuracy == results['SVM']:
    predictions = svm.run(FREQ_DIST_FILE, BI_FREQ_DIST_FILE, TRAIN_PROCESSED_DATASET, TEST_PROCESSED_DATASET, True)
elif max_accuracy == results['Naive Bayes']:
    predictions = naivebayes.run(FREQ_DIST_FILE, BI_FREQ_DIST_FILE, TRAIN_PROCESSED_DATASET, TEST_PROCESSED_DATASET, True)
else:
    predictions = logistic.run(FREQ_DIST_FILE, BI_FREQ_DIST_FILE, TRAIN_PROCESSED_DATASET, TEST_PROCESSED_DATASET, True)


negative = 0
positive = 0
for prediction in predictions:
    if int(prediction) > 0:
        positive += 1
    else:
        negative += 1

total = len(predictions)
positive = twitter_driver.percentage(positive, total)
negative = twitter_driver.percentage(negative, total)

positive = format(positive, ".2f")
negative = format(negative, ".2f")

labels = ['Positive ['+str(positive)+'%]', 'Negative ['+str(negative)+'%]']
sizes = [positive, negative]
colors = ['yellowgreen', 'red']
patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.legend(patches, labels, loc="best")
plt.title("How many people are reacting on " + searchTerm + " by analyzing " + str(noOfSearchTerms) + " Tweets.")
plt.axis('equal')
plt.tight_layout()
plt.show()