from __future__ import division
import psycopg2
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
import re
from nltk.tag import StanfordNERTagger
from nltk import pos_tag
from nltk.chunk import conlltags2tree
from nltk.tree import Tree
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sqlalchemy import create_engine
import datetime
import json


stanfordNLP_classifier_path = '/home/ubuntu/english.all.3class.distsim.crf.ser.gz'
stanfordNER_jar_path = '/home/ubuntu/stanford-ner-3.8.0.jar'
ram_usage = 16000
jvm_ram_setting = '-mx' + str(ram_usage) + 'm'

def update_index(high_index):
    """
    Updates low and high index
    """
    file = open("data.txt", "w")
    file.writelines(str(high_index+1) + "\n")
    file.writelines(str(high_index+501) + "\n")
    file.close()

def read_index():
    """
    We want to process 1000 articles at a time
    This reads the low id and high id from a text file and returns a dict that can be passed to the postgress query
    """
    raw_data = [int(line.rstrip('\n')) for line in open("data.txt")]
    print(raw_data)
    data = {}
    data['min'] = raw_data[0]
    data['max'] = raw_data[1]
    return data

'''
Remove stopwords from article body and append result to dataframe.
'''
def stopword_articles(df):
    stop_words = stopwords.words('english')
    stop_words = stop_words + [',', '.', '!', '?', '"','\'', '/', '\\', '-', '--', '—', '(', ')', '[', ']', '\'s', '\'t', '\'ve', '\'d', '\'ll', '\'re', 'image']
    stop_words = set(stop_words) # making this a set increases performance for large documents

    stopworded_body = []
    for body in df['tokenized_body']:
        stopworded_body.append([w.lower() for w in body if w not in stop_words])

    se = pd.Series(stopworded_body)
    df['stopworded_body'] = se.values

'''
Tokenize each article by word and do a simple total word count. Append results to dataframe.
'''
def tokenize_and_wordcount_articles(df):
    tokenized_body = []
    for body in df['body']:
        body = body.decode('utf-8')
        tokens = nltk.word_tokenize(body)
        tokenized_body.append(tokens)

    se = pd.Series(tokenized_body)
    df['tokenized_body'] = se.values

    word_count = []
    for body in df['tokenized_body']:
        word_count.append(len(body))

    se = pd.Series(word_count)
    df['word_count'] = se.values

'''
Lemmatize (get common root words) stopworded articles and append results to dataframe.
'''
def lemmatize_articles(df):
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

    se = pd.Series(lemmatized_body)
    df['lemmatized_body'] = se.values

'''
Perform bag of words (frequency distribution) calculation on each article and append results to dataframe.
'''
def bag_of_words_articles(df):
    word_bag = []
    for body in df['lemmatized_body']:
        fdist = FreqDist(body)
        # FreqDist returns a special nltk.probability.FreqDist type
        # This is a list of tuples
        word_bag.append(fdist.most_common())

    se = pd.Series(word_bag)
    df['word_bag'] = se.values

'''
Calculate the lexical diversity (a measure of the richness of vocabulary used)
of each article and append results to dataframe.
'''
def lexical_diversity(text):
    return len(set(text)) / len(text) * 100

def calculate_lexical_diversity(df):
    lex_div = []
    for body in df['stopworded_body']:
        lex_div.append(lexical_diversity(body))

    se = pd.Series(lex_div)
    df['lexical_diversity'] = se.values

'''
We need to tokenize at the sentence level for our sentiment analysis.
'''
def get_sentence_tokens(df_body_column):
    body_sentences = []
    for body in df_body_column:
        body = body.decode('utf-8')
        sentences = sent_tokenize(body)
        body_sentences.append(sentences)
    return body_sentences

'''
Our sentiment analysis process is based on find a sentiment analysis score for
each sentence in an article so to get the average score for an article
we need to divide each individual sum by the number of sentences in the article
'''
def average_sentiment(neg, neu, pos, compound, length):
    result = {}
    result['neg'] = neg/length
    result['neu'] = neu/length
    result['pos'] = pos/length
    result['compound'] = compound/length
    return result

'''
Calculate sentiment scores from each article and append the resulting list of dictionaries to the dataframe.
'''
def calculate_sentiment(df):
    body_sentences = get_sentence_tokens(df['body'])
    sid = SentimentIntensityAnalyzer()
    sentiment_score = []
    for text in body_sentences:
        neg = 0
        neu = 0
        pos = 0
        compound = 0
        try:
            for sent in text:
                ss = sid.polarity_scores(sent)
                neg += ss['neg']
                neu += ss['neu']
                pos += ss['pos']
                compound += ss['compound']
            result = average_sentiment(neg, neu, pos, compound, len(text))
            json_dict = json.dumps(result)
            sentiment_score.append(json_dict)
        except:
            sent_dict = {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
            json_dict = json.dumps(sent_dict)
            sentiment_score.append(json_dict)
    return sentiment_score
# pass in the list of sentiment scores for our articles to assign a binary positive or negative value
def assign_sentiment(score_list):
    binary_sentiment = []
    for score in score_list:
         score_dict = json.loads(score)
         if score_dict['compound'] > 0:
            binary_sentiment.append(1)
         else:
            binary_sentiment.append(0)
    return binary_sentiment

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

def stanfordNE2tree(ne_tagged_sent):
    bio_tagged_sent = stanfordNE2BIO(ne_tagged_sent)
    sent_tokens, sent_ne_tags = zip(*bio_tagged_sent)
    sent_pos_tags = [pos for token, pos in pos_tag(sent_tokens)]

    sent_conlltags = [(token, pos, ne) for token, pos, ne in zip(sent_tokens, sent_pos_tags, sent_ne_tags)]
    ne_tree = conlltags2tree(sent_conlltags)
    return ne_tree
'''
Extract named entities from each article and append to dataframe.
'''
def extract_named_entities(df, st):
    classified_texts = []
    for body in df['tokenized_body']:
        classified_texts.append(st.tag(body))

    ne_trees = []
    for text in classified_texts:
        try:
            ne_trees.append(stanfordNE2tree(text))
        except:
            ne_trees.append(' ')

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

    se = pd.Series(ne_in_sents)
    df['named_entities'] = se.values

def main():

    engine = create_engine('postgresql://postgres:secret@ec2-52-27-114-159.us-west-2.compute.amazonaws.com:9000/cap')
    st = StanfordNERTagger(stanfordNLP_classifier_path, stanfordNER_jar_path, encoding='utf-8', java_options=jvm_ram_setting)
    while True:
        data = read_index()
        if data['min'] > 160000:
            break
        print(data)
        conn = psycopg2.connect("dbname='cap' user='postgres' host='ec2-52-27-114-159.us-west-2.compute.amazonaws.com' port=9000 password ='secret'")
        df = pd.read_sql_query(sql="SELECT * FROM articles WHERE ID >=%(min)s and ID <= %(max)s ORDER BY ID", con=conn, params=data)

        tokenize_and_wordcount_articles(df)

        stopword_articles(df)

        lemmatize_articles(df)

        bag_of_words_articles(df)

        extract_named_entities(df, st)

        calculate_lexical_diversity(df)

        sentiment_score = calculate_sentiment(df)
        binary_sentiment = assign_sentiment(sentiment_score)

        se = pd.Series(sentiment_score)
        df['sentiment_score'] = se.values
        se = pd.Series(binary_sentiment)
        df['binary_sentiment'] = se.values

        engine = create_engine('postgresql://postgres:secret@ec2-52-27-114-159.us-west-2.compute.amazonaws.com:9000/cap')
        df.to_sql(name='nlp_dim_hpc', con=engine, if_exists='append')
        update_index(data['max'])
        print('completed')

if __name__ == '__main__':
    main()
