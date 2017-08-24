'''
Created on Aug 23, 2017

@author: Varela

motivation: nltk package exploration 

'''

s = "Albert Einstein was born on March 14, 1879"
tags = nltk.pos_tag(s.split())
print tags 

print nltk.ne_chunk(tags)
print nltk.ne_chunk(tags).draw()

s = "Steve Jobes was the CEO of Apple"
tags = nltk.pos_tag(s.split())
print tags 

print nltk.ne_chunk(tags)
print nltk.ne_chunk(tags).draw()


