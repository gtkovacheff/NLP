import numpy as np
import pandas as pd
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import os
import plotly.express as ex
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('vader_lexicon')
from sklearn.cluster import KMeans
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from wordcloud import WordCloud, STOPWORDS
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from nltk.util import ngrams
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import random
plt.rc('figure', figsize=(17, 13))

with open('pandasOptions.env', 'r') as f:
    for line in f:
        pd.set_option(line.split("=")[0], int(line.split("=")[1]))

data = pd.read_csv('./Data/covidvaccine.csv')
data.head(3)

#remove twitter handlers
data.text = data.text.apply(lambda x: re.sub('@[^\s]+', '', str(x)))
#remove hashtags
data.text = data.text.apply(lambda x: re.sub(r'\B#\S+', '', x))
#remove urls
data.text = data.text.apply(lambda x: re.sub(r'http\S+', '', x))
#remove all the special characters
data.text = data.text.apply(lambda x: ' '.join(re.findall(r'\w+', x)))
#remove all single chars
data.text = data.text.apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', '', x))
# Substituting multiple spaces with single space
data.text = data.text.apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))

sid = SIA()
data['sentiments'] = data['text'].apply(lambda x: sid.polarity_scores(' '.join(re.findall(r'\w+', x.lower()))))
data['Positive Sentiment'] = data['sentiments'].apply(lambda x: x['pos']+1*(10**-6))
data['Neutral Sentiment'] =  data['sentiments'].apply(lambda x: x['neu']+1*(10**-6))
data['Negative Sentiment'] = data['sentiments'].apply(lambda x: x['neg']+1*(10**-6))
data.drop(columns=['sentiments'],inplace=True)