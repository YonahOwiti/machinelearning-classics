'''
Created on Aug 23, 2017

@author: Varela

motivation: Sentiment analysis
POS tagging 
EX Bob is great ->(noun, verb, adjective)
full table			https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

'''

#POS tagging 
# Bob is great
# (noun, verb, adjective)

# text = word_tokenize("Machine learning is great")
print "Pos_tag('Machine learning is great'): ", nltk.pos_tag("Machine learning is great".split())
