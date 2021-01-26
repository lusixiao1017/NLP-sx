# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 18:23:04 2021

@author: Sixiao
"""

#%%
trained_model = "D:\\Project-website\GloVe model\glove.6B.100d.txt"

import numpy as np


def loadGloveModel(trained_model):
    print ("Loading Glove Model")
    with open(trained_model, encoding="utf8" ) as f:
        content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model




#%%
import re
from nltk.corpus import stopwords
import pandas as pd

#%%
def preprocess(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split 
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    cleaned_words = list(set([w for w in words if w not in stopword_set]))

    return cleaned_words


#%%
def cosine_distance_between_two_words(word1, word2):
    import scipy
    return (1- scipy.spatial.distance.cosine(model[word1], model[word2]))

#%%
def calculate_heat_matrix_for_two_sentences(s1,s2):
    s1 = preprocess(s1)
    s2 = preprocess(s2)
    result_list = [[cosine_distance_between_two_words(word1, word2) for word2 in s2] for word1 in s1]
    result_df = pd.DataFrame(result_list)
    result_df.columns = s2
    result_df.index = s1
    return result_df
#%%

def cosine_distance_wordembedding_method(s1, s2):
    import scipy
    vector_1 = np.mean([model[word] for word in preprocess(s1)],axis=0)
    vector_2 = np.mean([model[word] for word in preprocess(s2)],axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')
#%%
def heat_map_matrix_between_two_sentences(s1,s2):
    df = calculate_heat_matrix_for_two_sentences(s1,s2)
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5,5)) 
    ax_blue = sns.heatmap(df, cmap="YlGnBu")
    # ax_red = sns.heatmap(df)
    print(cosine_distance_wordembedding_method(s1, s2))
    return ax_blue

#%%
import time


ss1 = 'The president greets the press in Chicago'
ss2 = 'Obama speaks to the media in Illinois'

ss11 ='The teacher gave his speech to an empty room'
ss12 ='The teacher gave his speech to a full room'
ss22 ='There was almost nobody when the professor was talking'


Reference_answer = " The answer is B. According to the beer’s law (e to the power of –u) the attenuation coefficient can be from 0 to infinity, which 0 means no attenuation, and infinity means absorbs all the radiation. "      
Students_answer = " I choose B, because the attenuation coefficient does not limit to 0 to 100 percent. "

start = time.time()



model = loadGloveModel(trained_model)
heat_map_matrix_between_two_sentences(Students_answer,Reference_answer)

end = time.time()

print ('the loading time for the model',end-start)