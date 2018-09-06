
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
from numpy import nan
from bs4 import BeautifulSoup    
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from math import sqrt


# In[6]:


train_data = pd.read_csv('train.tsv',sep='\t')
test_data = pd.read_csv('test.tsv',sep='\t')


# In[10]:


Sentiment_words=[]
for row in train_data['Sentiment']:
    if row ==0:
        Sentiment_words.append('negative')
    elif row == 1:
        Sentiment_words.append('neutral')
    elif row == 2:
        Sentiment_words.append('somewhat negative')
    elif row == 3:
        Sentiment_words.append('somewhat positive')
    elif row == 4:
        Sentiment_words.append('positive')
    else:
        Sentiment_words.append('Failed')
train_data['Sentiment_words'] = Sentiment_words


# In[98]:


word_count=pd.value_counts(train_data['Sentiment_words'].values, sort=False)


# In[97]:


Index = [1,2,3,4,5]
plt.figure(figsize=(15,5))
plt.bar(Index,word_count,color = 'blue')
plt.xticks(Index,['negative','neutral','somewhat negative','somewhat positive','positive'],rotation=45)
plt.ylabel('word_count')
plt.xlabel('word')
plt.title('Count of Moods')
plt.bar(Index, word_count)
for a,b in zip(Index, word_count):
    plt.text(a, b, str(b) ,color='green', fontweight='bold')


# In[14]:


def review_to_words(raw_review): 
    review =raw_review
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    return (' '.join(review))


# In[16]:


import nltk
nltk.download('wordnet')


# In[17]:


corpus= []
for i in range(0, 156060):
    corpus.append(review_to_words(train_data['Phrase'][i]))


# In[19]:


corpus


# In[12]:


train_data.head()


# In[22]:


corpus1= []
for i in range(0, 66292):
    corpus1.append(review_to_words(test_data['Phrase'][i]))


# In[23]:


train_data['new_Phrase']=corpus


# In[24]:


train_data.drop(['Phrase'],axis=1,inplace=True)


# In[25]:


positive=train_data[train_data['Sentiment_words']==('positive')]


# In[26]:


positive


# In[27]:


words = ' '.join(positive['new_Phrase'])
split_word = " ".join([word for word in words.split()])


# In[34]:


wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(split_word)
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[35]:


pos=positive['new_Phrase']


# In[37]:


vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2000)


# In[38]:


pos_words = vectorizer.fit_transform(pos)
pos_words = pos_words.toarray()
pos= vectorizer.get_feature_names()


# In[51]:


dist = np.sum(pos_words, axis=0)
for tag, count in zip(pos, dist):
    print (tag,count)


# In[53]:


postive_new= pd.DataFrame(dist)


# In[55]:


postive_new.columns=['word_count']


# In[56]:


postive_new['word'] = pd.Series(pos, index=postive_new.index)


# In[58]:


postive_new1=postive_new[['word','word_count']]


# In[59]:


postive_new1.head()


# In[60]:


top_30_words=postive_new1.sort_values(['word_count'],ascending=[0])


# In[61]:


top_30_words.head(30)


# In[62]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x__train = cv.fit_transform(corpus).toarray()
x__test= cv.fit_transform(corpus1).toarray()
y = train_data.iloc[:, 2].values


# In[73]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x__train, y, test_size = 0.40, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_real_pred = classifier.predict(x__test)


# In[90]:


test_data['Sentiment'] = y_real_pred


# In[92]:


Sentiment_words=[]
for row in test_data['Sentiment']:
    if row ==0:
        Sentiment_words.append('negative')
    elif row == 1:
        Sentiment_words.append('neutral')
    elif row == 2:
        Sentiment_words.append('somewhat negative')
    elif row == 3:
        Sentiment_words.append('somewhat positive')
    elif row == 4:
        Sentiment_words.append('positive')
    else:
        Sentiment_words.append('Failed')
test_data['Sentiment_words'] = Sentiment_words


# In[95]:


test_data.to_csv('result.csv', sep='\t')


# In[86]:


mse = ((y_pred - y_test) ** 2).mean()


# In[87]:


rmse = sqrt(mse)


# In[89]:


rmse

