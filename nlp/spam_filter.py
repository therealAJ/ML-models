
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import string

from nltk.corpus import stopwords

get_ipython().magic('matplotlib inline')

#nltk.download_shell()


# #### *loading and exploring the data*

messages = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',
                      names=['label','message'])

messages.head()
messages.describe()
messages.info()

messages.groupby('label').describe()

messages['length'].plot.hist(bins=30)
messages.hist(column='length',by='label',bins=50,figsize=(12,3))


# #### *Text pre-processing*

def text_process(mess):
    """
    1. remove punctuation
    2. remove stop words
    3. return list of clean text words
    """
    
    nopunc = [char for char in mess if char not in string.punctuation]
    
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower not in stopwords.words('english')] 


from sklearn.feature_extraction.text import CountVectorizer 

bag_of_words_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
messages_bow = bag_of_words_transformer.transform(messages['message'])

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)

messages_tfidf = tfidf_transformer.transform(messages_bow)

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])

# #### *Train Test Split*

from sklearn.model_selection import train_test_split

msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],messages['label'],test_size=0.3)

# #### *Data Pipeline*

from sklearn.pipeline import Pipeline


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])

pipeline.fit(msg_train,label_train)

predictions = pipeline.predict(msg_test)

# #### *Evaluating the model*

from sklearn.metrics import classification_report

print(classification_report(label_test,predictions))
