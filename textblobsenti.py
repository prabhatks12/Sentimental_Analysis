# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 23:25:16 2018

@author: prabhat
"""

from textblob import TextBlob
print("Enter any statment")
wiki = TextBlob("I am very sad")
wiki.tags
wiki.words
wiki.sentiment.polarity
