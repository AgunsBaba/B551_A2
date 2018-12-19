#!/bin/python
# I originally was calculating probabilites for the words, locations and words given location before realizing that
# if I was trying to maximize the numerator, I didn't need the probability for the words, which saved me
# a lot of computation. I also ended up stripping all words of punctuation and capitalization so that there wouldn't
# be a distinction between people capitalizing words or adding hash-tags or other punctuation. My classifier did
# seem to put all the tweets in the test set in the same location, I tested different thresholds for including
# a word in the probabilities but that only changed which location it returned not the fact that it returned the same
# location for every tweet. I ended up multiplying the probabilities returned by a very large number as I couldn't
# find anything on how many decimals python could compare to, but this still didn't change the fact that I got the same
# location every time. I also checked the actual probabilities being returned and they were all different numbers but
# one location was always consistently higher than the others, which seemed suggest that my program was at least
# returning the correct location in that it was the one with the highest probability. In the end, I couldn't find any
# reason for it to always return the same location, as I checked all my probability dictionaries and they all looked
# like what I would expect, which made me wonder if it was something about the training data, perhaps that tweets
# from certain locations tended to be longer and contain more words, raising the probability of that location given
# the words in the tweet. 

import re
import sys
import numpy
import string

train = sys.argv[1]
test = sys.argv[2]
output = sys.argv[3]

l = re.compile(r'_[A-Z]')

tweets = []
locations = []
unique_locations = []
loc_words = {}
with open(train, 'r') as f:
    for line in f:
        tweets.append(line)

for tweet in tweets:
    array = tweet.split(' ')
    locations.append(array[0])
    if array[0] not in unique_locations:
        unique_locations.append(array[0])
        loc_words[array[0]] = []
    for i in range(1,len(array)):
        word = array[i].translate(None, string.punctuation).lower()
        loc_words[array[0]].append(word)


loc_prob= {}
word_loc = {}


# Probabilities of a given location P(l)
total_loc = len(locations)
for location in unique_locations:
    count = locations.count(location)
    loc_prob[location] = float(count)/total_loc


#Probabilities of word given location P(w|l)
for k in range(0, len(unique_locations)):
    word_loc[unique_locations[k]] = {}
    uniq_loc_words = set(loc_words[unique_locations[k]])
    for word in uniq_loc_words:
        if loc_words[unique_locations[k]].count(word) > 15:
            word_loc[unique_locations[k]][word] = float(loc_words[unique_locations[k]].count(word))/len(loc_words[unique_locations[k]])


def bayes_solver(location, words):
    p_location = loc_prob[location]
    p_words_loc = []
    for word in words:
        word1 = word.translate(None, string.punctuation)
        if word1 in word_loc[location]:
            p_words_loc.append(word_loc[location][word1])
    p_location *=numpy.prod(p_words_loc)
    return float(p_location)*(10**60)


with open(test, 'r') as f2:
    with open(output, 'a') as f3:
        for line in f2:
            array = line.split(' ')
            for i in range(1,len(array)):
                array[i] = array[i].lower()
            tweet = " ".join(array[1:])
            max = 0
            answer = ''
            for location in unique_locations:
                a = bayes_solver(location, array[1:])
                if a > max:
                    max = a
                    answer = location
            f3.write(answer+" "+array[0]+" "+tweet)










