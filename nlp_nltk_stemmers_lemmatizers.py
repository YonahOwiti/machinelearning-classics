'''
Created on Aug 23, 2017

@author: Varela

motivation: nltk package exploration

Stemming and lemmatizing
Both reduce the word to a base form.
Stemmers only truncate
Lemmatizers adjust to the right radical
stem('wolves')  -> wolv
lemmatize('wolves')  -> wolf
 

'''

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

#Stemmer reduces to base form by truncation
stemmer = PorterStemmer()

print "Wolves stems to :", stemmer.stem('Wolves')
print "Jumping stems to :", stemmer.stem('Jumping')

#Lemmatizers reduces to base form with correct radical
lemmatizer = WordNetLemmatizer()

print "Wolves lemmatizes to :", lemmatizer.lemmatize('Wolves') #--> should be working
print "Jumping lemmatizes to :", lemmatizer.lemmatize('Jumping')
