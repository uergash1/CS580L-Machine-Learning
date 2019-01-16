# import naivebayes
# import svm
# import decisiontree
# import logistic
# import neuralnet
# print("Naive Bayes:", naivebayes.run())
# print("SVM:", svm.run())
# print("decisiontree:", decisiontree.run())
# print("logistic:", logistic.run())
# print("neuralnet:", neuralnet.run())

# results = {}
# results['Naive Bayes'] = 17
#
# results['SVM'] = 18
#
# results['Decision Tree'] = 19
#
# results['Logistic Regression'] = 20
#
# results['Neural Network'] = 20.5
#
# print(max(list(results.values())))

import preprocess
import twitter_driver
import utils

# searchTerm = input("Enter keyword/hashtag to search about: ")
# noOfSearchTerms = int(input("Enter how many tweets to analyze: "))
#
# tweets = twitter_driver.get_tweets(searchTerm, noOfSearchTerms)
# # for tw in tweets:
# #     print(tw.text)
# tweets_csv = utils.save_tweets(tweets)
# preprocess.run(tweets_csv, test_file=True)


# def run():
#     t = 1
#     q = 2
#     return t, q
#
# t1, t2 = run()
#
# print(t1, t2)

clean_text = []
with open("data/test-small-processed.csv", 'r') as csv:
    lines = csv.readlines()
    for line in lines:
        number, sentiment, text = line.split(',')
        clean_text.append(text)



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=4, max_features = 10000)
vz = vectorizer.fit_transform(clean_text)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

from sklearn.cluster import MiniBatchKMeans

num_clusters = 2
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
kmeans = kmeans_model.fit(vz)
kmeans_clusters = kmeans.predict(vz)
kmeans_distances = kmeans.transform(vz)
sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(num_clusters):
    print("Cluster %d:" % i)
    for j in sorted_centroids[i, :5]:
        print(' %s' % terms[j])
    print()
