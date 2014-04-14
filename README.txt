README

The following are the 2 files that contain code
train.py
test.py

To run the code,
Please execute the train.py, and enter the path of the training file when prompted for it
This will train the system and store the model files

To test data please execute test.py which will use the stored models for classification

Various statistics like time taken to train,vocabulary length,memory usage and Prior Probabilities based on training data are printed on the console

Some of the important data structures and variables worth mentioning are:
1)freq_dic
A dictionary that adds all the features 

2)pos_feature_dic that includes all the words occurring in positively classified reviews in the training data along with their total frequency counts. 

3) neg_feature_dic that includes all the words occurring in negatively classified reviews in the training data along with their total frequency counts

Methods:
removePunct: takes an input a list of words and removes the punctuations using regular expressions

removeStopWords:takes as input a list of words and removes the standerd list of “English” stop words  in the NLTK but preserves the words “not”, “nor” and “no”

Variables:
Flags are used to skip over the first line of the CSV which contains headings
pos_review_count and neg_review_count are used to record the number of positive and negative reviews. This is used to calculate the prior probabilities

Imports needed:(Standard libraries along with NLTK stemmer are used. Following is a list of import statements which is already added to the .py scripts

import nltk
import csv
import re
import sys
import math
import operator
import collections
from math import log
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import time