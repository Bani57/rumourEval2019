import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from os.path import isfile
import glob
import json
import html
from ast import literal_eval
import _pickle as c_pickle
import datetime
import pytz
from emoji.unicode_codes import UNICODE_EMOJI
import re
from gensim.models import word2vec
import logging
from sklearn.cluster import KMeans
from nltk.corpus import words as nltk_dictionary
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from nltk import FreqDist
from nltk import TextCollection
from nltk import pos_tag
from nltk import download
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import networkx as nx
from collections import Counter
from random import sample
from scipy.spatial.distance import cosine
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from scipy.stats.kde import gaussian_kde
