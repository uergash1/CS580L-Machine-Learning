# from textblob import TextBlob
import sys, tweepy
import matplotlib.pyplot as plt

def percentage(part, whole):
    return 100 * float(part) / float(whole)

def get_tweets(searchTerm, noOfSearchTerms):
    consumerKey = "C4PtIkLXDo8lNYdQa9ESnxuTw"
    consumerSecret = "sf8DUbXeOegcXhJjMImqMuGyZYXVBrCYpAaKGPGIVAfOCPJMoq"
    accessToken = "825570200393232386-jTxI97H9MOiuTD6e5yZH05I0IFO60ud"
    accessTokenSecret = "KHW18Xn2TfdhPxlmfUwLi2fselIT6NgMGgjCR6kmP8Jxm"

    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)

    tweets = tweepy.Cursor(api.search, q=searchTerm).items(noOfSearchTerms)
    return tweets

# positive = 0
# negative = 0
# neutral = 0
# polarity = 0
#
# for tweet in tweets:
#     analysis = TextBlob(tweet.text)
#     polarity += analysis.sentiment.polarity
#
#     if analysis.sentiment.polarity == 0:
#         neutral += 1
#     elif analysis.sentiment.polarity < 0.00:
#         negative += 1
#     elif analysis.sentiment.polarity > 0.00:
#         positive += 1
#
# positive = percentage(positive, noOfSearchTerms)
# negative = percentage(negative, noOfSearchTerms)
# neutral = percentage(neutral, noOfSearchTerms)
# polarity = percentage(polarity, noOfSearchTerms)
#
# positive = format(positive, ".2f")
# negative = format(negative, ".2f")
# neutral = format(neutral, ".2f")
#
# print("How many people are reacting on " + searchTerm + " by analyzing " + str(noOfSearchTerms) + " Tweets.")
#
# if polarity == 0:
#     print("Neutral")
# elif polarity < 0.00:
#     print("Negative")
# elif polarity > 0.00:
#     print("Positive")
#
# labels = ['Positive ['+str(positive)+'%]', 'Neutral ['+str(neutral)+'%]', 'Negative ['+str(negative)+'%]']
# sizes = [positive, neutral, negative]
# colors = ['yellowgreen', 'gold', 'red']
# patches, texts = plt.pie(sizes, colors=colors, startangle=90)
# plt.legend(patches, labels, loc="best")
# plt.title("How many people are reacting on " + searchTerm + " by analyzing " + str(noOfSearchTerms) + " Tweets.")
# plt.axis('equal')
# plt.tight_layout()
# plt.show()