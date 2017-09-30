
# coding: utf-8

# In[1]:

import psycopg2
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
import re
from __future__ import division
from nltk.tag import StanfordNERTagger


# In[2]:

conn = psycopg2.connect("dbname='cap' user='postgres' host='ec2-35-163-99-253.us-west-2.compute.amazonaws.com' port=9000 password ='secret'")
df = pd.read_sql_query("SELECT * FROM articles limit 5", conn)


# In[3]:

tokenized_body = []
for body in df['body']:
    body = body.decode('utf-8')
    tokens = nltk.word_tokenize(body)
    tokenized_body.append(tokens)


# In[4]:

se = pd.Series(tokenized_body)
df['tokenized_body'] = se.values


# In[19]:

word_count = []
for body in df['tokenized_body']:
    word_count.append(len(body))


# In[20]:

se = pd.Series(word_count)
df['word_count'] = se.values


# In[21]:

stop_words = stopwords.words('english')
stop_words = stop_words + [',', '.', '!', '?', '"','\'', '/', '\\', '-', '--', '—', '(', ')', '[', ']', '\'s', '\'t', '\'ve', '\'d', '\'ll', '\'re']
stop_words = set(stop_words) # making this a set increases performance for large documents


# In[34]:

df


# In[22]:

stopworded_body = []
for body in df['tokenized_body']:
    stopworded_body.append([w.lower() for w in body if w not in stop_words])


# In[23]:

se = pd.Series(stopworded_body)
df['stopworded_body'] = se.values


# In[24]:

wnl = nltk.WordNetLemmatizer()
lemmatized_words = []
lemmatized_body = []
for body in df['stopworded_body']:
    # We need to tag words with their parts of speech before the WordNet lemmatizer will work properly
    pos_tagged_body = nltk.pos_tag(body)
    lemmatized_words = []
    for word, tag in pos_tagged_body:
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        if not wntag:
            lemma = word
        else:
            lemma = wnl.lemmatize(word, wntag)
        lemmatized_words.append(lemma)
    lemmatized_body.append(lemmatized_words)


# In[25]:

se = pd.Series(lemmatized_body)
df['lemmatized_body'] = se.values


# In[26]:

word_bag = []
for body in df['lemmatized_body']:
    fdist = FreqDist(body)
    # FreqDist returns a special nltk.probability.FreqDist type
    # This is a list of tuples
    # Here is an example of how to access the elements for future reference
#     print(fdist.most_common())
    # Access an individual tuple
#     print(fdist.most_common()[0])
    # Access the word from the tuple
#     print(fdist.most_common()[0][0])
    # Access the count from the tuple
#     print(fdist.most_common()[0][1])
    # Append to list as ordered frequency distribution
    word_bag.append(fdist.most_common())


# In[27]:

se = pd.Series(word_bag)
df['word_bag'] = se.values


# In[28]:

st = StanfordNERTagger('/media/justin/Data/Google Drive/Assignments and Projects/Machine Learning/NLP/english.all.3class.distsim.crf.ser.gz',
					   '/media/justin/Data/Google Drive/Assignments and Projects/Machine Learning/NLP/stanford-ner.jar',
					   encoding='utf-8')


# In[29]:

classified_texts = []
for body in df['tokenized_body']:
    classified_texts.append(st.tag(body))

# print(classified_text)


# In[30]:

from nltk import pos_tag
from nltk.chunk import conlltags2tree
from nltk.tree import Tree

def stanfordNE2BIO(tagged_sent):
    bio_tagged_sent = []
    prev_tag = "O"
    for token, tag in tagged_sent:
        if tag == "O": #O
            bio_tagged_sent.append((token, tag))
            prev_tag = tag
            continue
        if tag != "O" and prev_tag == "O": # Begin NE
            bio_tagged_sent.append((token, "B-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag == tag: # Inside NE
            bio_tagged_sent.append((token, "I-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag != tag: # Adjacent NE
            bio_tagged_sent.append((token, "B-"+tag))
            prev_tag = tag

    return bio_tagged_sent


# In[31]:

def stanfordNE2tree(ne_tagged_sent):
    bio_tagged_sent = stanfordNE2BIO(ne_tagged_sent)
    sent_tokens, sent_ne_tags = zip(*bio_tagged_sent)
    sent_pos_tags = [pos for token, pos in pos_tag(sent_tokens)]

    sent_conlltags = [(token, pos, ne) for token, pos, ne in zip(sent_tokens, sent_pos_tags, sent_ne_tags)]
    ne_tree = conlltags2tree(sent_conlltags)
    return ne_tree


# In[40]:

ne_trees = []
for text in classified_texts:
    try:
        ne_trees.append(stanfordNE2tree(text))
    except:
        ne_trees.append(' ')


# In[41]:

ne_in_sent = []
ne_in_sents = []
for tree in ne_trees:
    ne_in_sent = []
    for subtree in tree:
        if type(subtree) == Tree: # If subtree is a noun chunk, i.e. NE != "O"
            ne_label = subtree.label()
            ne_string = " ".join([token for token, pos in subtree.leaves()])
            ne_in_sent.append((ne_string, ne_label))
    ne_in_sents.append(ne_in_sent)


# In[42]:

se = pd.Series(ne_in_sents)
df['named_entities'] = se.values


# In[43]:

def lexical_diversity(text):
    return len(set(text)) / len(text) * 100


# In[44]:

lex_div = []
for body in df['stopworded_body']:
    lex_div.append(lexical_diversity(body))


# In[45]:

se = pd.Series(lex_div)
df['lexical_diversity'] = se.values


# In[48]:

get_ipython().system(u'jupyter nbconvert --to script /home/justin/GitHub/CapstoneI/config_template.ipynb')


# In[ ]:



