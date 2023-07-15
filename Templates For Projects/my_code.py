# Import necessary libraries
import pandas as pd
import numpy as np
import datetime as dt
from functools import reduce
import sys
import my_code as mc
import string
import time
import re
import math
import os

# Viz libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
import plotly.graph_objects as go


# ML libraries
from scipy import stats
import statsmodels.api as sm
import sklearn as sk
from sklearn.model_selection import train_test_split
from tempfile import mkdtemp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile, f_regression, f_classif
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
# from fbprophet import Prophet
# from fbprophet.plot import plot_plotly, add_changepoints_to_plot
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from scipy.sparse import hstack
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords 

# Geo libraries
from geopy.geocoders import Nominatim

# # Optional Libraries
# from collections import defaultdict
# from datetime import timedelta  
# from dateutil.relativedelta import relativedelta
# import functools
from IPython.display import display, Markdown

# Initialize styling params
plt.rcParams['figure.figsize'] = (8.0, 6.0) #setting figure size
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (9, 6)

# sns.set_style("whitegrid")
sns.set_style("darkgrid")
sns.set_context("notebook")
sns.set_palette("viridis")

pd.set_option("display.max_columns", 50)
pd.set_option('display.max_colwidth', 1000)
pd.plotting.register_matplotlib_converters()
os.environ["PYTHONHASHSEED"] = "123"



def get_coordinates(address):
    """
    Given an address, use the geopy library to get the latitude and longitude.
    
    Args:
    address (str): The address to geocode.
    
    Returns:
    tuple: The latitude and longitude, or raises a ValueError if the address was not found.
    """
    geolocator = Nominatim(user_agent="my-app")  # Create a Nominatim geolocator with a custom user_agent
    location = geolocator.geocode(address)  # Try to geocode the address
    if location:
        return (location.latitude, location.longitude)
    else:
        print("Error: Address not found")
        return False
        

def get_mapping_lat_long(addresses):
    """
    Given a list of addresses, get the latitude and longitude for each address.
    
    Args:
    addresses (list): The list of addresses to geocode.
    
    Returns:
    dict: A dictionary mapping addresses to (latitude, longitude) tuples.
    """
    # Get the coordinates for each address
    coordinates_list = list(map(get_coordinates, addresses))
    lat_lon_dict = dict(zip(addresses, coordinates_list))  # Create a dictionary mapping addresses to coordinates
    
    # Print the results
    for address, coordinates in lat_lon_dict.items():
        if coordinates:
            latitude, longitude = coordinates
            print(f"Address: {address}")
            print(f"Latitude: {latitude}")
            print(f"Longitude: {longitude}")
            print()
        else:
            print(f"Address not found or coordinates not available: {address}")
            print()
            
    return lat_lon_dict

def add_latitude_and_longitude(df, mappings, key_column, lat_column, long_column):
    """
    Add latitude and longitude to a dataframe based on an existing column of addresses.
    
    Args:
    df (DataFrame): The dataframe to add latitude and longitude to.
    mappings (dict): A dictionary mapping addresses to (latitude, longitude) tuples.
    key_column (str): The column of the dataframe that contains the addresses.
    lat_column (str): The name of the new latitude column.
    long_column (str): The name of the new longitude column.
    """
    # Create dictionaries mapping addresses to latitudes and longitudes
    latitudes = {k: v[0] for k, v in mappings.items() if v}

    longitudes = {k: v[1] for k, v in mappings.items() if v}

    # Map the latitudes and longitudes to the dataframe
    df[lat_column] = df[key_column].map(latitudes).combine_first(df[lat_column])

    df[long_column] = df[key_column].map(longitudes).combine_first(df[long_column])
    return df

def one_hot_encode(df, col, drop_col=False, drop_first=False):
    """
    One-hot encode a column of a dataframe.
    
    Args:
    df (DataFrame): The dataframe to encode.
    col (str): The column to encode.
    drop_col (bool): Whether to drop the original column.
    drop_first (bool): Whether to drop the first level of the encoded data.
    
    Returns:
    DataFrame: The dataframe with the encoded column.
    """
    # Create the one-hot encoded dataframe
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first).astype(int)
    df = pd.concat([df, dummies], axis=1)
    if drop_col:
        df = df.drop(columns=col)  # Drop the originalcolumn if requested
    return df


def numeric_columns_assessment(df):
    """
    Function to assess numeric columns in a DataFrame.
    
    Parameters:
    df (DataFrame): The DataFrame to assess.
    
    Returns:
    DataFrame: A DataFrame with statistics for each numeric column.
    """
    # Select numeric columns
    numeric_columns = df.select_dtypes(include=[np.number, np.datetime64])
    
    # Calculate statistics
    numeric_means = numeric_columns.mean().values
    numeric_medians = numeric_columns.median().values
    numeric_mins_maxs = list(zip(numeric_columns.min().values, numeric_columns.max().values))
    numeric_STD = numeric_columns.std().values
    numeric_skew = numeric_columns.skew().values
    numeric_dtypes = numeric_columns.dtypes.values


    # Return a DataFrame with the statistics
    return pd.DataFrame(index=['Numeric Columns', 'Mean', 'Median', 'Range', 'STD', 'Skew', 'Dtype'], data=[numeric_columns.columns.tolist(), 
                numeric_means, numeric_medians, numeric_mins_maxs, numeric_STD, numeric_skew, numeric_dtypes]).T

def non_numeric_columns_assessment(df):
    """
    Function to assess non-numeric columns in a DataFrame.
    
    Parameters:
    df (DataFrame): The DataFrame to assess.
    
    Returns:
    DataFrame: A DataFrame with statistics for each non-numeric column.
    """
    # Select non-numeric columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number])
    
    # Calculate statistics
    uniques_non = non_numeric_columns.nunique().values
    most_common_non = list(zip([non_numeric_columns[i].value_counts().index[0] for i in non_numeric_columns.columns], 
                            [non_numeric_columns[i].value_counts()[0] for i in non_numeric_columns.columns]))
    least_common_non = list(zip([non_numeric_columns[i].value_counts().index[-1] for i in non_numeric_columns.columns], 
                            [non_numeric_columns[i].value_counts()[-1] for i in non_numeric_columns.columns]))

    # Return a DataFrame with the statistics
    return pd.DataFrame(index=['Non-Numeric Columns', '# Uniques', 'Most Common', 'Least Common'], 
                        data=[non_numeric_columns.columns.tolist(), uniques_non, most_common_non, least_common_non]).T

# setup for tokenizer
# stemmer = nltk.stem.PorterStemmer()

# # create the custom tokenizer function from class notebook
# ENGLISH_STOP_WORDS = stopwords.words('english')

# def my_tokenizer(sentence):
#     """
#     Function to tokenize a sentence.
    
#     Parameters:
#     sentence (str): The sentence to tokenize.
    
#     Returns:
#     list: A list of tokens.
#     """
#     # remove punctuation and set to lower case
#     for punctuation_mark in string.punctuation:
#         sentence = sentence.replace(punctuation_mark,'').lower()

#     # split sentence into words
#     listofwords = sentence.split(' ')
#     listofstemmed_words = []
    
#     # remove stopwords and any tokens that are just empty strings
#     for word in listofwords:
#         if (not word in ENGLISH_STOP_WORDS) and (word!=''):
#             # Stem words
#             stemmed_word = stemmer.stem(word)
#             listofstemmed_words.append(stemmed_word)

#     return listofstemmed_words

# def coverage(X, tokenizer, vectorizer):
#     """
#     Function to compute the coverage of a vectorizer's features over the tokens in a text.
    
#     Parameters:
#     X (Series): The text to analyze.
#     tokenizer (function): The tokenizer function to use.
#     vectorizer (Vectorizer): The vectorizer to use.
    
#     Returns:
#     float: The coverage.
#     """
#     # Tokenize the text using your tokenizer
#     tokens = X.apply(tokenizer)

#     # Flatten the list of tokens and compute the number of unique tokens
#     unique_tokens = set(token for tokens in tokens for token in tokens)

#     # Get the 1-grams from the vectorizer's feature names
#     vectorizer_features = set(feature for feature in vectorizer.get_feature_names_out() if ' ' not in feature)

#     # Compute the number of features in your vectorizer that are also in the unique tokens
#     common_features = unique_tokens.intersection(vectorizer_features)

#     # Compute the coverage
#     coverage = len(common_features) / len(unique_tokens)
    
#     print(f"The coverage for this corpus is {coverage * 100:.2f}%")

#     return coverage



print("Versions used in this notebook:")
print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")
print(f"Seaborn version: {sns.__version__}")
print(f"Matplotlib version: {mpl.__version__}")
# print(f"Scipy version: {scipy.__version__}")
# print(f"Statsmodels version: {sm.__version__}")
print(f"SKLearn version: {sk.__version__}")
