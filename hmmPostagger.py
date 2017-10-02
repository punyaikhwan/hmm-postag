
#    Source code merupakan hasil adaptasi pada webiste 
#    http://www.katrinerk.com/courses/python-worksheets/hidden-markov-models-for-pos-tagging-in-python

# This HMM addresses the problem of part-of-speech tagging. It estimates
# the probability of a tag sequence for a given word sequence as follows:
#
# Say words = w1....wN
# and tags = t1..tN
#
# then
# P(tags | words) is_proportional_to  product P(ti | t{i-1}) P(wi | ti)
#
# To find the best tag sequence for a given sequence of words,
# we want to find the tag sequence that has the maximum P(tags | words)

from __future__ import print_function
from __future__ import division
import nltk
import sys
import fileinput
import sys
import codecs
import os.path
import logging
import re
import file_util
import traceback
from file_util import ID,FORM,CPOSTAG #column index for the columns we'll need
try:
    import argparse
except:
    #we are on Python 2.6 or older
    from compat import argparse

def makeCorpus(trees):
    print("Creating corpus...")
    global corpus
    corpus = []
    for comments,tree in trees:
        sentence = []
        #tree adalah satu sentence
        #trees adalah sekumpulan sentence
        for line in tree:
            if not line[ID].isdigit():
                continue
            word = line[FORM]
            tag = line[CPOSTAG]
            sentence.append((word, tag))
        corpus.append(sentence)
    print("Corpus created.")

def makeModel(percentageSplit):
    brown_tags_words = [ ]
    cutoff = int(len(corpus)*percentageSplit/100)
    tagged_data = corpus[:cutoff]
    for sent in tagged_data:
        # sent is a list of word/tag pairs
        # add START/START at the beginning
        brown_tags_words.append( ("START", "START") )
        # then all the tag/word pairs for the word/tag pairs in the sentence.
        # shorten tags to 2 characters each
        brown_tags_words.extend([ (tag, word) for (word, tag) in sent ])
        # then END/END
        brown_tags_words.append( ("END", "END") )
    # conditional frequency distribution
    global cfd_tagwords
    cfd_tagwords = nltk.ConditionalFreqDist(brown_tags_words)
    # conditional probability distribution
    global cpd_tagwords
    cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)

    # Estimating P(ti | t{i-1}) from corpus data using Maximum Likelihood Estimation (MLE):
    # P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})
    brown_tags = [tag for (tag, word) in brown_tags_words ]

    # make conditional frequency distribution:
    # count(t{i-1} ti)
    global cfd_tags
    cfd_tags= nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
    # make conditional probability distribution, using
    # maximum likelihood estimate:
    # P(ti | t{i-1})
    global cpd_tags
    cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)

    #####
    # Viterbi:
    # If we have a word sequence, what is the best tag sequence?
    #
    # The method above lets us determine the probability for a single tag sequence.
    # But in order to find the best tag sequence, we need the probability
    # for _all_ tag sequence.
    # What Viterbi gives us is just a good way of computing all those many probabilities
    # as fast as possible.

    # what is the list of all tags?
    global distinct_tags
    distinct_tags = set(brown_tags)

def tagSentence(sentence):
    sentlen = len(sentence)

    # viterbi:
    # for each step i in 1 .. sentlen,
    # store a dictionary
    # that maps each tag X
    # to the probability of the best tag sequence of length i that ends in X
    global viterbi
    viterbi = [ ]

    # backpointer:
    # for each step i in 1..sentlen,
    # store a dictionary
    # that maps each tag X
    # to the previous tag in the best tag sequence of length i that ends in X
    global backpointer
    backpointer = [ ]

    first_viterbi = { }
    first_backpointer = { }
    for tag in distinct_tags:
        # don't record anything for the START tag
        if tag == "START": continue
        first_viterbi[ tag ] = cpd_tags["START"].prob(tag) * cpd_tagwords[tag].prob( sentence[0] )
        first_backpointer[ tag ] = "START"
        
    viterbi.append(first_viterbi)
    backpointer.append(first_backpointer)

    best_tag = []
    currbest = max(first_viterbi.keys(), key = lambda tag: first_viterbi[ tag ])
    best_tag.append(currbest)

    for wordindex in range(1, len(sentence)):
        this_viterbi = { }
        this_backpointer = { }
        prev_viterbi = viterbi[-1]
        
        for tag in distinct_tags:
            # don't record anything for the START tag
            if tag == "START": continue

            # if this tag is X and the current word is w, then 
            # find the previous tag Y such that
            # the best tag sequence that ends in X
            # actually ends in Y X
            # that is, the Y that maximizes
            # prev_viterbi[ Y ] * P(X | Y) * P( w | X)
            # The following command has the same notation
            # that you saw in the sorted() command.
            best_previous = max(prev_viterbi.keys(),
                                key = lambda prevtag: \
                prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex]))

            # Instead, we can also use the following longer code:
            # best_previous = None
            # best_prob = 0.0
            # for prevtag in distinct_tags:
            #    prob = prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex])
            #    if prob > best_prob:
            #        best_previous= prevtag
            #        best_prob = prob
            #
            this_viterbi[ tag ] = prev_viterbi[ best_previous] * \
                cpd_tags[ best_previous ].prob(tag) * cpd_tagwords[ tag].prob(sentence[wordindex])
            this_backpointer[ tag ] = best_previous

        currbest = max(this_viterbi.keys(), key = lambda tag: this_viterbi[ tag ])
        best_tag.append(currbest)
        # print( "Word", "'" + sentence[ wordindex].encode('UTF8') + "'", "current best two-tag sequence:", this_backpointer[ currbest], currbest)
        # print( "Word", "'" + sentence[ wordindex] + "'", "current best tag:", currbest)


        # done with all tags in this iteration
        # so store the current viterbi step
        viterbi.append(this_viterbi)
        backpointer.append(this_backpointer)

    return best_tag

def findBestTagSequence(sentence):
    sentlen = len(sentence)

    # viterbi:
    # for each step i in 1 .. sentlen,
    # store a dictionary
    # that maps each tag X
    # to the probability of the best tag sequence of length i that ends in X
    global viterbi
    viterbi = [ ]

    # backpointer:
    # for each step i in 1..sentlen,
    # store a dictionary
    # that maps each tag X
    # to the previous tag in the best tag sequence of length i that ends in X
    global backpointer
    backpointer = [ ]

    first_viterbi = { }
    first_backpointer = { }
    for tag in distinct_tags:
        # don't record anything for the START tag
        if tag == "START": continue
        first_viterbi[ tag ] = cpd_tags["START"].prob(tag) * cpd_tagwords[tag].prob( sentence[0] )
        first_backpointer[ tag ] = "START"
        
    viterbi.append(first_viterbi)
    backpointer.append(first_backpointer)

    currbest = max(first_viterbi.keys(), key = lambda tag: first_viterbi[ tag ])

    for wordindex in range(1, len(sentence)):
        this_viterbi = { }
        this_backpointer = { }
        prev_viterbi = viterbi[-1]
        
        for tag in distinct_tags:
            # don't record anything for the START tag
            if tag == "START": continue

            # if this tag is X and the current word is w, then 
            # find the previous tag Y such that
            # the best tag sequence that ends in X
            # actually ends in Y X
            # that is, the Y that maximizes
            # prev_viterbi[ Y ] * P(X | Y) * P( w | X)
            # The following command has the same notation
            # that you saw in the sorted() command.
            best_previous = max(prev_viterbi.keys(),
                                key = lambda prevtag: \
                prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex]))

            # Instead, we can also use the following longer code:
            # best_previous = None
            # best_prob = 0.0
            # for prevtag in distinct_tags:
            #    prob = prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex])
            #    if prob > best_prob:
            #        best_previous= prevtag
            #        best_prob = prob
            #
            this_viterbi[ tag ] = prev_viterbi[ best_previous] * \
                cpd_tags[ best_previous ].prob(tag) * cpd_tagwords[ tag].prob(sentence[wordindex])
            this_backpointer[ tag ] = best_previous

        currbest = max(this_viterbi.keys(), key = lambda tag: this_viterbi[ tag ])
        # print( "Word", "'" + sentence[ wordindex].encode('UTF8') + "'", "current best two-tag sequence:", this_backpointer[ currbest], currbest)
        # print( "Word", "'" + sentence[ wordindex] + "'", "current best tag:", currbest)


        # done with all tags in this iteration
        # so store the current viterbi step
        viterbi.append(this_viterbi)
        backpointer.append(this_backpointer)

    # done with all words in the sentence.
    # now find the probability of each tag
    # to have "END" as the next tag,
    # and use that to find the overall best sequence
    prev_viterbi = viterbi[-1]
    best_previous = max(prev_viterbi.keys(),
                        key = lambda prevtag: prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob("END"))

    prob_tagsequence = prev_viterbi[ best_previous ] * cpd_tags[ best_previous].prob("END")

    # best tagsequence: we store this in reverse for now, will invert later
    best_tagsequence = [best_previous ]
    # invert the list of backpointers
    backpointer.reverse()

    # go backwards through the list of backpointers
    # (or in this case forward, because we have inverter the backpointer list)
    # in each case:
    # the following best tag is the one listed under
    # the backpointer for the current best tag
    current_best_tag = best_previous
    for bp in backpointer:
        best_tagsequence.append(bp[current_best_tag])
        current_best_tag = bp[current_best_tag]

    best_tagsequence.reverse()
    # print( "The best tag sequence is:", end = " ")
    # for t in best_tagsequence: print (t, end = " ")
    # print("\n")
    # print( "The probability of the best tag sequence is:", prob_tagsequence)
    return best_tagsequence

def getAccuracyBestTag(percetageSplit):
    print("---ACCURACY USING BEST TAG---")
    cutoff = int(len(corpus)*percetageSplit/100)
    numTruetag = 0
    numWords = 0
    test_data = corpus[cutoff:]
    for sent in test_data:
        #process for being sentence
        words = []
        for wt in sent:
            words.append(wt[0])
            numWords += 1

        #postagging and counting
        tags = tagSentence(words)
        i = 0
        for wt in sent:
            if wt[1] == tags[i]:
                numTruetag += 1
            i +=1
    print("Truly tagged: ",numTruetag)
    print("Number words: :",numWords)
    print("Accuracy: ", numTruetag/numWords)

def getAccuracyBestSeqTag(percetageSplit):
    print("---ACCURACY USING BEST SEQUENCE TAG---")
    cutoff = int(len(corpus)*percetageSplit/100)
    numTruetag = 0
    numWords = 0
    test_data = corpus[cutoff:]
    for sent in test_data:
        #process for being sentence
        words = []
        for wt in sent:
            words.append(wt[0])
            numWords += 1

        #postagging and counting
        tags = findBestTagSequence(words)
        i = 1
        for wt in sent:
            if wt[1] == tags[i]:
                numTruetag += 1
            i +=1
    print("Truly tagged: ",numTruetag)
    print("Number words: :",numWords)
    print("Accuracy: ", numTruetag/numWords)

opt_parser = argparse.ArgumentParser(description="CoNLL-U validation script")

io_group=opt_parser.add_argument_group("Input / output options")
opt_parser.add_argument('input', nargs='?', help='Input file name, or "-" or nothing for standard input.')
opt_parser.add_argument('output', nargs='?', help='Output file name, or "-" or nothing for standard output.')
args = opt_parser.parse_args() #Parsed command-line arguments
inp, out=file_util.in_out(args)
trees = file_util.trees(inp)
trees = list(trees)

makeCorpus(trees)
makeModel(100)
# getAccuracyBestTag(25)
# getAccuracyBestSeqTag(25)
print("Tag kalimat: Ahli rekayasa optik mendesain komponen dari instrumen optik seperti lensa.")
print("Tag menggunakan best tag: ", tagSentence((" Ahli rekayasa optik mendesain komponen dari instrumen optik seperti lensa").split()))
print("Tag menggunakan tag seq: ", findBestTagSequence((" Ahli rekayasa optik mendesain komponen dari instrumen optik seperti lensa").split())[1:])
