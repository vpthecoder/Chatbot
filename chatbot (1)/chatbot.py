import pickle
import random
import string

import nltk
from nltk import *
# import libraries
import pathlib
import sys
import re
import pickle
from random import seed
from random import randint
import nltk
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

knowledgebase = open('kbase.txt', 'r')
knowledge = knowledgebase.read()

knowledge = knowledge.lower()
sent_tokens = sent_tokenize(knowledge)
word_tokens = word_tokenize(knowledge)

tokens = [t for t in word_tokens if t not in stopwords.words('english')]

# use lemmatizer
wnl = WordNetLemmatizer()
lemmas = [wnl.lemmatize(t) for t in tokens]
lemma = ' '.join(lemmas)
lemmas = sent_tokenize(lemma)

lemmer = WordNetLemmatizer()


def get_response(response):
    jordanbot = ''
    lemmas.append(response)
    k = TfidfVectorizer()
    l = k.fit_transform(lemmas)
    vals = cosine_similarity(l[-1], l)
    flat = vals.flatten()
    idx = vals.argsort()[0][-2]
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        jordanbot = jordanbot + "Be more specific please."
        return jordanbot
    else:
        jordanbot = jordanbot + sent_tokens[idx]
        return jordanbot


def chatbot(user_dict, comments, question_answer):
    print("If you do not want to chat, type exit. Else, type anything else.")
    chatter = input()
    if chatter == 'exit':
        print("Bye.")
        exit()


    asking = 1
    print("I am Michael Jordan's number one fan. Ask questions about him. Type exit to exit this conversation")
    print("What is your name?")
    
    name = input()
    if name in user_dict:
        print("Welcome back!")
    while (asking == 1):
        print("Ask a question!")
        question = input()
        if question in question_answer.keys():
            print("You asked this already!")
        else:
            question_answer.update()
            question = question.lower()
            if question != 'exit':
                if (question == 'thanks' or question == 'thank you'):
                    asking = 0
                    print("You are welcome. Ask another question.")
                else:

                    answer = (get_response(question))
                    print(answer)
                    question_answer.update({question: answer})
                    lemmas.remove(question)
            else:
                asking = 0
                comments.append(question_answer)
                user_dict.update({name: comments})
                print("Thank you for chatting with me!")
                chatbot(user_dict, comments, question_answer)


def main():
    chatbot({}, [], {})


if __name__ == '__main__':
    main()
