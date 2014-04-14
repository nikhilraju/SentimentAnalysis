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

# Remove Punctuations
def removePunctuation(input_list):
    punctuation=re.compile(r'[,./?!":;|-]')
    
    punct_remove = [punctuation.sub(" ", word) for word in input_list]
    
    return punct_remove

# Remove Stop Words
def removeStopWords(input_list):
    stop=stopwords.words('english')
    #Need to preserve not,nor and no as these contain information
    stop.remove('not')
    stop.remove('nor')
    stop.remove('no')
    stop.append('')
    
    for s in stop:
        while s in input_list:
            input_list.remove(s)
    return input_list

#Variable Declarations
stemmer=PorterStemmer()
pos_feature_dic={}
neg_feature_dic={}
freq_dic={}
pos_review_count=0
neg_review_count=0
training_example_count=0
flag=False
chisquared_dic={}

#Reading the Reviews from Training Data and Building Feature Dictionary
start=time.time()
print 'Please enter the Path of the training file'
training_file_path=raw_input()

with open (training_file_path,'rb') as training_file:
    training_data=csv.reader(training_file)
    
    for row in training_data:
        #Using this flag to skip the first row in the csv
        if flag==False:
            flag=True
        else:
            training_example_count+=1
            #print 'Word list after splitting',word_list_split
            word_list_split=re.split('\s+',row[1].lower())           
            #print 'Word list after removing punctuations',word_list_minus_punct
            word_list_minus_punct=removePunctuation(word_list_split)
            #print 'Word list after removing Stop Words',word_list_minus_stop       
            word_list_minus_stop=removeStopWords(word_list_minus_punct)
            words_with_count = collections.Counter(word_list_minus_stop)
            if int(row[0])==1:
                pos_review_count+=1
                for word in words_with_count:
                    stemmed_word=word
                    #stemmed_word =stemmer.stem(word)      
                    if stemmed_word in freq_dic:
                        current_value = freq_dic[stemmed_word]
                        current_value[0] += words_with_count[stemmed_word]
                        current_value[1] += 1
                        freq_dic[stemmed_word] = current_value
                    else:
                        freq_dic[stemmed_word] = [words_with_count[stemmed_word], 1]
                    if stemmed_word in pos_feature_dic:
                        current_value = pos_feature_dic[stemmed_word]
                        current_value[0] += words_with_count[stemmed_word]
                        current_value[1] += 1
                        pos_feature_dic[stemmed_word] = current_value
                    else:
                        pos_feature_dic[stemmed_word]= [words_with_count[stemmed_word], 1]
                        
            else: 
                neg_review_count+=1
                for word in words_with_count:
                    stemmed_word=word
                    #stemmed_word = stemmer.stem(word)
                    if stemmed_word in freq_dic:
                        current_value = freq_dic[stemmed_word]
                        current_value[0] += words_with_count[stemmed_word]
                        current_value[1] += 1
                        freq_dic[stemmed_word] = current_value
                    else:
                        freq_dic[stemmed_word] = [words_with_count[stemmed_word], 1]    
                    if stemmed_word in neg_feature_dic:
                        current_value = neg_feature_dic[stemmed_word]
                        current_value[0] += words_with_count[stemmed_word]
                        current_value[1] += 1                    
                        neg_feature_dic[stemmed_word] = current_value
                    else:
                        neg_feature_dic[stemmed_word]= [words_with_count[stemmed_word], 1]
                        
#Feature Selection, Compute Chi-Squared test statistic            
for feature in freq_dic:
    if feature in pos_feature_dic:        
        f11=pos_feature_dic[feature][1]
    else:
        f11=0
    if feature in neg_feature_dic:
        f10=neg_feature_dic[feature][1]
    else:
        f10=0
    f01=pos_review_count-f11
    f00=neg_review_count-f10

    chisquared_dic[feature]=((float) (((f11+f10+f01+f00)*pow((f11*f00-f10*f01),2))/(float)((f11+f01)*(f11+f10)*(f10+f00*f01+f00))))

#Sort Vocab dictionary in descending order of Chi-Squared Statistic
sorted_chisquared_dic=sorted(chisquared_dic.iteritems(),key=operator.itemgetter(1),reverse=True)
pos_prior=(float)(pos_review_count)/(training_example_count)
neg_prior=(float)(neg_review_count)/(training_example_count)


token_list=[]

output_list=[]
total_test_pos_reviews=0
total_test_neg_reviews=0
totalfreqpos=0
totalfreqneg = 0
total_vocab=len(freq_dic)

#Set the value of k to tune the number of features selected
k=3000

# Code to delete the unimportant features from the positive and negative dictionaries
imp_feature_list=[]

for [key,val] in sorted_chisquared_dic:
    imp_feature_list.append(key)
    if(len(imp_feature_list)==k):
        break

#Delete other features from the positve and negative dictionaries using above list
print 'Lengths of pos,neg dictionaries before feature extraciton',len(pos_feature_dic),len(neg_feature_dic)
print 'Length of the entire feature '
for key in pos_feature_dic.keys():
    if key not in imp_feature_list:
        pos_feature_dic.pop(key)
        if key in freq_dic:
            freq_dic.pop(key)


for key in neg_feature_dic.keys():
    if key not in imp_feature_list:
        neg_feature_dic.pop(key)
        if key in freq_dic:
            freq_dic.pop(key)

print 'Lengths of pos,neg dictionaries after feature extraction',len(pos_feature_dic),len(neg_feature_dic)

# Exporting the model files positive dictionary,negative dictionary and Vocab
f=open('pos_dictionary.txt','w')
pickle.dump(pos_feature_dic,f)
f.close()
  
f=open('neg_dictionary.txt','w')
pickle.dump(neg_feature_dic,f)
f.close()
  
f=open('vocabulary.txt','w')
pickle.dump(freq_dic,f)
f.close()

print 'TRAINING COMPLETED,STATISTICS'
print 'The time taken to train was:',time.time()-start
print 'The memory used by dictionaries is as follows:'
print 'The vocabulary contains',total_vocab,'words and uses',sys.getsizeof(freq_dic),'bytes'
print 'Positive Features Dictionary',len(pos_feature_dic),'features and uses',sys.getsizeof(pos_feature_dic),'bytes'
print 'Negative Features Dictionary',len(neg_feature_dic),'features and uses',sys.getsizeof(neg_feature_dic),'bytes'
print 'Chi-Squared Values',len(freq_dic),'and uses',sys.getsizeof(chisquared_dic),'bytes'
print'PRIORS',pos_prior,neg_prior