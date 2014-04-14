import pickle
import csv
import re
from train import removePunctuation  
from train import removeStopWords
from train import pos_prior
from train import neg_prior
from train import sorted_chisquared_dic
import math
import time

#Variable Declarations
flag=False
testing_example_count=0
token_list=[]
output_list=[]
total_test_pos_reviews=0
total_test_neg_reviews=0
totalfreqpos=0
totalfreqneg = 0

#Import the trained model files
f=open('pos_dictionary.txt','r')
pos_feature_dic=pickle.load(f)

f=open('neg_dictionary.txt','r')
neg_feature_dic=pickle.load(f)

f=open('vocabulary.txt','r')
freq_dic=pickle.load(f)
 
print 'Please enter the Path of the testing file'
testing_file_path=raw_input()
#Classification 
start=time.time()
with open ('test.csv','rb') as testing_file:
    testing_data=csv.reader(testing_file)
    for row in testing_data:
        token_pos_prob_list=[]
        token_neg_prob_list=[]
        outputList=[]
        if flag==False:
            flag=True
            i=0
        else:
            testing_example_count+=1
            word_list_split=re.split('\s+',row[0].lower())
            word_list_minus_punct=removePunctuation(word_list_split)
            word_list_minus_stop=removeStopWords(word_list_minus_punct)
            for word in word_list_minus_stop:
                token_list.append(word)
                if word in pos_feature_dic:
                    pos_prob_token=(float)(pos_feature_dic[word][0]+1)/(len(freq_dic)+totalfreqpos)
                    token_pos_prob_list.append(math.log(pos_prob_token))
                else:
                    token_pos_prob_list.append(math.log((float)(1)/(len(freq_dic)+totalfreqpos)))
                if word in neg_feature_dic:
                    neg_prob_token=(float)(neg_feature_dic[word][0]+1)/(len(freq_dic)+totalfreqneg)
                    token_neg_prob_list.append(math.log(neg_prob_token))
                else:
                    token_neg_prob_list.append(math.log((float)(1)/(len(freq_dic)+totalfreqneg)))

            pos_prob_total=0
            pos_prob_total = reduce(lambda a,d: a + d,token_pos_prob_list)
            pos_prob_total = pos_prob_total + math.log(pos_prior)
            neg_prob_total = math.log(neg_prior) + (reduce(lambda a, d:a + d,token_neg_prob_list))
            if pos_prob_total>neg_prob_total:
                review=1
                total_test_pos_reviews+=1
            else:
                review=0
                total_test_neg_reviews+=1
            i+=1
            output_list.append([i, review])
    c = csv.writer(open("result.csv", "wb"))
    c.writerow(['Id', 'Category'])
    for row in output_list:
        c.writerow(row)
# print 'SORTED CHI-SQUARED VALUE DICTIONARY IS',sorted_chisquared_dic
# 
#      
# print 'POSITIVE DICTIONARY IS'
# for key,value in pos_feature_dic.iteritems():
#     print key,value
# 
# print 'NEGATIVE DICTIONARY IS'
# for key,value in neg_feature_dic.iteritems():
#     print key,value

print 'Positive: ',total_test_pos_reviews,'Negative: ',total_test_neg_reviews
print 'Testing Completed...The classified results have been stored in the result.csv file'    
print 'The time taken to train was:',time.time()-start
