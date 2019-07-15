#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import nltk
nltk.download('punkt')
import nltk
nltk.download('averaged_perceptron_tagger')
import nltk
nltk.download('tagsets')
import nltk
nltk.download('wordnet')
import nltk
nltk.download('movie_reviews')


# In[6]:


#this is how Naive Bayes classifier expects the input
def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict 
create_word_features(["the","quick","brown","quick", "a", "fox"])


# In[7]:


neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append((create_word_features(words),"negative"))
print(neg_reviews[0])
print(len(neg_reviews))


# In[8]:


pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append((create_word_features(words),"positive"))
#print(pos_reviews[0])
print(len(pos_reviews))


# In[11]:


train_set = neg_reviews[:750] +  pos_reviews[:750]
test_set = neg_reviews[750:] + pos_reviews[750:]
print(len(train_set), len(test_set))


# In[12]:


#train the classifier
classifier = NaiveBayesClassifier.train(train_set)


# In[14]:


#find accuracy percentage
accuracy = nltk.classify.util.accuracy(classifier,test_set)
print(accuracy * 100)


# In[15]:


review_santa = '''
I stumbled upon this heaping pile of garbage by googling, “Worst holiday movies ever,” and props to Netflix for actually having it available so that I could torture myself for 81 excruciating minutes. Made in 1964, SCCtM was directed by Nicholas Webster, who is perhaps best known for directing the iconic 1977 Bigfoot episode of “In Search Of…”. The majority of the movie’s actors were only known for this train wreck of a Christmas movie, which is the most depressing thing I can think of. But not all the actors are no-names. Feast your eyes on 10-year-old Martian girl and 80s 

'''
print(review_santa)


# In[16]:


words = word_tokenize(review_santa)
words = create_word_features(words)

#predict if the word is pos or neg
classifier.classify(words)


# In[19]:


review_spirit = '''
An absolutely masterful movie. I feel like the Spirit Realm that the main character Chihiro goes too is visually stunning to look at. The character designs, the buildings - it's all so unique. During the movie I also felt really bad for Chihiro - how she had to go through all these things during the film (if you saw it, you might understand). I liked when Haku and Chihiro were together too. They just had such a nice relationship together. Also, the train sequence with Chihiro and No - Face was excellent. I just sat there still in amazement wondering why other animated films can't have these types of scenes in them. Also, there were some scenes that just really got me hooked (well, the whole movie did) and got me excited for what was gonna happen next. There were also some scenes that were kinda disturbing to watch. Well, What can I say. The animation is beautiful, the characters are unforgettable, the story is great. This has definitely turned to one of my favorite animated movies of all time.
'''
print(review_spirit)


# In[21]:


words = word_tokenize(review_spirit)
words = create_word_features(words)

#predict if the word is pos or neg
classifier.classify(words)


# In[ ]:




