# Import necessary libraries

import pandas as pd
import numpy as np
import datetime as dt
from functools import reduce
import sys
import string
import time
import re
import math
import os
from IPython.display import display, Markdown
from summarytools import dfSummary, tabset



# Viz libraries

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import month_plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.tsa.seasonal as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ----------------------------------------------------------------------------------------------------------------------------

# ML libraries

import scipy.stats as stats
import scipy
import statsmodels.api as sm
import sklearn as sk
from tempfile import mkdtemp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.experimental import enable_halving_search_cv    # noqa
from sklearn.model_selection import GridSearchCV, train_test_split, HalvingGridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectPercentile, f_regression, f_classif
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, silhouette_score
from sklearn import metrics
from mlxtend.plotting import plot_decision_regions

# # ----------------------------------------------------------------------------------------------------------------------------

# # ML Models
# # Basic Classifier/Regression Models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


# # ----------------------------------------------------------------------------------------------------------------------------

# # Advanced Ensemble Models

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
# # from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


# # ----------------------------------------------------------------------------------------------------------------------------

# # Clustering Models

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture


# # ----------------------------------------------------------------------------------------------------------------------------

# # Time Series Models

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
# from prophet import plot_plotly, add_changepoints_to_plot
# from keras.models import Sequential
# from keras.layers import LSTM


# # ----------------------------------------------------------------------------------------------------------------------------

# ML NLP libraries

# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from scipy.sparse import hstack
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords 


# # ----------------------------------------------------------------------------------------------------------------------------

# Geo libraries

from geopy.geocoders import Nominatim


# # ----------------------------------------------------------------------------------------------------------------------------


# # Optional Libraries

# from collections import defaultdict
# from dateutil.relativedelta import relativedelta

# ----------------------------------------------------------------------------------------------------------------------------

# Initialize styling params
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (9, 6)
# plt.style.use('fivethirtyeight')

# sns.set_style("darkgrid")
# sns.set_style("whitegrid")
sns.set_palette("viridis")
sns.set_context("notebook")

pd.set_option("display.max_columns", 50)
pd.set_option('display.max_colwidth', 1000)
pd.plotting.register_matplotlib_converters()
os.environ["PYTHONHASHSEED"] = "42"
os.environ['KMP_DUPLICATE_LIB_OK']='True'



# ----------------------------------------------------------------------------------------------------------------------------
# Assessment Functions

# Original Numerical Data
def visualize_all_numeric(train_df):
    train_df_numeric = train_df.select_dtypes(include=[np.number])
    number_of_plots = int(np.ceil(len(train_df_numeric.columns)/3))

    # Plot creation
    fig, axs = plt.subplots(number_of_plots, 3, figsize=(15, number_of_plots *5))

    # Iterating through columns of interest
    for idx, column_name in enumerate(train_df_numeric.columns):
        row = idx // 3
        col = idx % 3
        plot_titles = column_name.replace("_", " ").title()
        
        sns.histplot(data=train_df_numeric, x=column_name, bins=20, kde=True, ax=axs[row, col])
        axs[row, col].grid(axis='y', linestyle='--', alpha=0.5)
        axs[row, col].set_title(f'Distribution of {plot_titles}')
        axs[row, col].set_xlabel(f'{plot_titles}')
        
        axs[row, col].axvline(train_df_numeric[column_name].mean(), color='r', linestyle='--', label='mean')
        axs[row, col].axvline(train_df_numeric[column_name].median(), color='g', linestyle='--', label='median')

        axs[row, col].legend()
    
    plt.suptitle('Distribution of Numerical Columns', fontsize=16)
    plt.tight_layout()
    



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


# ----------------------------------------------------------------------------------------------------------------------------
# Hyperparameter Functions and Piping


def evaluate_regression_model(model, X_test, y_test, plot=True):
    """
    Evaluate the performance of a linear model.

    Parameters:
    model (Model): Trained model for making predictions
    X_test (DataFrame): Features from the test set
    y_test (Series/DataFrame): Target variable from the test set

    Returns:
    scores (Series): Series containing performance scores
    residuals (Series): Residuals of the model
    """    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = metrics.r2_score(y_test, y_pred)
    adj_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    explained_variance = metrics.explained_variance_score(y_test, y_pred)
    
    # Put metrics into a pandas series
    scores = pd.Series({
        'R^2': r2,
        'Adjusted R^2': adj_r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'Explained Variance': explained_variance,
    })
    
    if plot:
        display(scores.to_frame('Scores').style.background_gradient(cmap='coolwarm'))

    return scores


def plot_regression_residuals(model, X_test, y_test):
    """
    Plot residuals of a linear model.

    Parameters:
    residuals (Series/DataFrame): Residuals of the model

    Returns:
    None
    """  
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Create Q-Q plot
    plt.figure(figsize=(12, 6))
    sm.qqplot(residuals, line='s')
    plt.title('Q-Q Plot')
    plt.show()

    # Create histogram of residuals
    sns.histplot(residuals, kde=True)
    plt.title('Histogram of Residuals')
    plt.show()

    # Create scatter plot of residuals
    plt.scatter(y_pred, residuals)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')
    plt.show()

    # Display plot
    plt.tight_layout()




def evaluate_classifier_model(best_model, X_test, y_test, confusion_matrix=True):
    """
    Evaluate the prediction performance of a given model on a test set.

    Parameters:
    best_model (Model): Trained model for making predictions
    X_test (DataFrame): Features from the test set
    y_test (Series/DataFrame): Target variable from the test set

    Returns:
    None
    """    
    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = str(np.round(accuracy_score(y_test, y_pred)*100, 1))

    # Display accuracy with Markdown
    display(Markdown(f"### Accuracy: \n The model's accuracy is **{accuracy}%**"))
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Convert to DataFrame
    report_df = pd.DataFrame(report).drop(['accuracy'], axis=1).T

    # Display styled DataFrame
    display(report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall','f1-score']))
    
    if confusion_matrix:
        """
        Plot a confusion matrix for a given model and test data.

        Parameters:
        best_model (Model): Trained model for making predictions
        X_test (DataFrame): Features from the test set
        y_test (Series/DataFrame): Target variable from the test set
        class_labels (list): List of class labels

        Returns:
        None
        """
        # Generate confusion matrix
        ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, 
                                        display_labels=range(len(y_test.unique())), 
                                        normalize='all', cmap='Blues', values_format='.2%')

        # Add title and axis labels
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Fix Grid
        plt.grid(False)

        # Show the plot
        plt.show()
        
    return report_df



def plot_classifier_residuals(model, X_test, y_test, aggregate=False):
    # Compute predicted probabilities
    y_pred_proba = model.predict_proba(X_test)

    residuals = []

    # compute residuals for each class
    for i in range(y_pred_proba.shape[1]):
        # Treat each class as binary outcome
        y_test_binary = (y_test == i).astype(int)
        
        # Compute residuals
        residuals_i = y_test_binary - y_pred_proba[:, i]
        residuals.append(residuals_i)
        
    residuals = np.array(residuals)

    if aggregate:
        # Aggregate residuals across all classes
        residuals_agg = residuals.sum(axis=0)

        # Plot aggregated residuals
        plt.scatter(y_pred_proba.sum(axis=1), residuals_agg, alpha=0.1)
        plt.xlabel('Fitted values (aggregated)')
        plt.ylabel('Residuals (aggregated)')
        plt.title('Residuals vs Fitted values (aggregated)')
        plt.show()

        # Plot aggregated residuals distribution
        plt.hist(residuals_agg, alpha=0.5)
        plt.xlabel('Residuals (aggregated)')
        plt.title('Residuals Distribution (aggregated)')
        plt.show()

        # Perform Durbin-Watson test
        dw = durbin_watson(residuals_agg)
        print('Durbin-Watson statistic (aggregated):', dw)

    else:
        # Plot residuals for each class
        for i in range(residuals.shape[0]):
            plt.scatter(y_pred_proba[:, i], residuals[i, :], alpha=0.1)
            plt.xlabel('Fitted values for class '+str(i))
            plt.ylabel('Residuals')
            plt.title('Residuals vs Fitted values for class '+str(i))
            plt.show()

            # Plot residuals distribution
            plt.hist(residuals[i, :], alpha=0.5)
            plt.xlabel('Residuals for class '+str(i))
            plt.title('Residuals Distribution for class '+str(i))
            plt.show()

            # Perform Durbin-Watson test
            dw = durbin_watson(residuals[i, :])
            print('Durbin-Watson statistic for class '+str(i)+':', dw)



def top_3_model_results(grid_outcomes, grid, show_bars=True):
    # Sort the DataFrame based on the mean test score and select the top 3
    top3 = grid_outcomes.sort_values('rank_test_score').head(3)

    # Pivot the DataFrame so the models are the columns and the parameters/scores are the rows
    top3 = top3.set_index('rank_test_score')

    # Filter the columns based on the regex pattern
    filtered_columns = top3.filter(regex=r'^param_', axis=1).columns

    # Parameters of the best model
    results1 = top3.loc[:, filtered_columns].sort_index().to_dict()


    # Append the other results to the list
    results2 = {
        'score_method': grid.scoring,
        'train_score_average': top3.loc[: , 'mean_train_score'],
        'validation_score_average': top3.loc[:, 'mean_test_score'],
        'n_splits': grid.n_splits_,
        'mean_fit_time': top3.loc[:, 'mean_fit_time'],
        'mean_score_time': top3.loc[:, 'mean_score_time']
    }

    results1.update(results2)

    # Convert the results to a DataFrame
    top_3_model_results = pd.DataFrame(results1)

    if show_bars:
        top_3_model_results.plot(kind='bar', y=['train_score_average', 'validation_score_average'], figsize=(10, 6), color=['blue', 'red'])

        # Set the y limits
        plt.ylim(top_3_model_results.validation_score_average.min() - 0.01, top_3_model_results.train_score_average.max() + 0.01)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.title('Comparison of Top Three Models on Training and Validation Scores', fontsize=16)

    return top_3_model_results


# Define a function to plot hyperparameters
def plot_average_score_of_hyperparameters(grid_outcomes, first_hyperparameter, second_hyperparameter=None, score_plot_name='Score'):
    # Group the grid outcomes by the first hyperparameter
    grouped_grid = grid_outcomes.groupby(first_hyperparameter)
    variable_plot_name= first_hyperparameter.replace('param_', '').replace('__', ' ').title()
    
    # Get the hyperparameter value of the model with the highest test score
    try: 
        winner = grid_outcomes.loc[grid_outcomes.mean_test_score.idxmax(), first_hyperparameter]
    except: 
        winner = grid_outcomes.loc[grid_outcomes.mean_test_score.argmax(), first_hyperparameter]

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot the mean test scores with error bars
    plt.errorbar(
        x=sorted(grouped_grid['mean_test_score'].mean().index),
        y=grouped_grid['mean_test_score'].mean().sort_index(),
        yerr=grouped_grid['std_test_score'].mean().sort_index(),
        marker='o',
        linestyle='-',
        color='red',
        label='Validation Scores'
    )

    # Plot the mean train scores with error bars
    plt.errorbar(
        x=sorted(grouped_grid['mean_train_score'].mean().index),
        y=grouped_grid['mean_train_score'].mean().sort_index(),
        yerr=grouped_grid['std_train_score'].mean().sort_index(),
        marker='o',
        linestyle='-',
        color='blue',
        label='Training Scores'
    )

    # If a second hyperparameter is provided
    if second_hyperparameter:
        # Plot the mean test scores with Seaborn lineplot, with line style varying by the second hyperparameter
        sns.lineplot(
            x=grid_outcomes[first_hyperparameter].sort_index(),
            y=grid_outcomes['mean_test_score'].sort_index(),
            style=grid_outcomes[second_hyperparameter].sort_index(),
            color='red',
            legend=False,
            alpha=0.3,
            errorbar=None
        )
        
        # Plot the mean train scores with Seaborn lineplot, with line style varying by the second hyperparameter
        sns.lineplot(
            x=grid_outcomes[first_hyperparameter].sort_index(),
            y=grid_outcomes['mean_train_score'].sort_index(),
            style=grid_outcomes[second_hyperparameter].sort_index(),
            color='blue',
            legend=True,
            alpha=0.3,
            errorbar=None
        )

    # Label the x-axis
    plt.xlabel(f'{variable_plot_name}')
    plt.xticks(rotation=90)

    # Draw a vertical line at the best model
    plt.axvline(winner, color='green', linestyle=':', linewidth=3, label='Best Model')

    # Label the y-axis
    plt.ylabel(f'Mean {score_plot_name}')

    # Add a title to the plot
    plt.title(f'Grid Search: {score_plot_name} vs {variable_plot_name}')

    # If a second hyperparameter is provided, place the legend outside the plot
    if second_hyperparameter:
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    else:
        plt.legend()
        


# Define a function to plot hyperparameters
def plot_average_time_of_hyperparameters(grid_outcomes, first_hyperparameter, second_hyperparameter=None):
    # Group the grid outcomes by the first hyperparameter
    grouped_grid = grid_outcomes.groupby(first_hyperparameter)
    variable_plot_name= first_hyperparameter.replace('param_', '').replace('__', ' ').title()

    try: 
        # Get the hyperparameter value of the model with the highest test score
        mean_fit_winner = grid_outcomes.loc[grid_outcomes.mean_fit_time.mean().idxmin(), first_hyperparameter]
        mean_score_winner = grid_outcomes.loc[grid_outcomes.mean_score_time.mean().idxmin(), first_hyperparameter]
    except: 
        # Get the hyperparameter value of the model with the highest test score
        mean_fit_winner = grid_outcomes.loc[grid_outcomes.mean_fit_time.mean().argmin(), first_hyperparameter]
        mean_score_winner = grid_outcomes.loc[grid_outcomes.mean_score_time.mean().argmin(), first_hyperparameter]

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot the mean test scores with error bars
    plt.errorbar(
        x=sorted(grouped_grid['mean_fit_time'].mean().index),
        y=grouped_grid['mean_fit_time'].mean().sort_index(),
        yerr=grouped_grid['std_fit_time'].mean().sort_index(),
        marker='o',
        linestyle='-',
        color='red',
        label='mean_fit_time'
    )

    # Plot the mean train scores with error bars
    plt.errorbar(
        x=sorted(grouped_grid['mean_score_time'].mean().index),
        y=grouped_grid['mean_score_time'].mean().sort_index(),
        yerr=grouped_grid['std_score_time'].mean().sort_index(),
        marker='o',
        linestyle='-',
        color='blue',
        label='mean_score_time'
    )

    # If a second hyperparameter is provided
    if second_hyperparameter:
        # Plot the mean test scores with Seaborn lineplot, with line style varying by the second hyperparameter
        sns.lineplot(
            x=grid_outcomes[first_hyperparameter].sort_index(),
            y=grid_outcomes['mean_fit_time'].sort_index(),
            style=grid_outcomes[second_hyperparameter].sort_index(),
            color='red',
            legend=False,
            alpha=0.3,
            errorbar=None
        )
        
        # Plot the mean train scores with Seaborn lineplot, with line style varying by the second hyperparameter
        sns.lineplot(
            x=grid_outcomes[first_hyperparameter].sort_index(),
            y=grid_outcomes['mean_score_time'].sort_index(),
            style=grid_outcomes[second_hyperparameter].sort_index(),
            color='blue',
            legend=True,
            alpha=0.3,
            errorbar=None
        )

    # Label the x-axis
    plt.xlabel(f'{variable_plot_name}')
    plt.xticks(rotation=90, fontsize=9)
    
    # Draw a vertical line at the best model
    plt.axvline(mean_fit_winner, color='red', linestyle=':', linewidth=3, label='Best Fit Time')
    plt.axvline(mean_score_winner, color='blue', linestyle=':', linewidth=3, label='Best Score Time')

    # Label the y-axis
    plt.ylabel(f'Mean Time (in Seconds)')

    # Add a title to the plot
    plt.title(f'Grid Search: Time (in Seconds) vs {variable_plot_name}')

    # If a second hyperparameter is provided, place the legend outside the plot
    if second_hyperparameter:
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    else:
        plt.legend()




# ----------------------------------------------------------------------------------------------------------------------------
# NLP Functions


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




# ----------------------------------------------------------------------------------------------------------------------------
# Geolocator Functions



# def get_coordinates(address):
#     """
#     Given an address, use the geopy library to get the latitude and longitude.
    
#     Args:
#     address (str): The address to geocode.
    
#     Returns:
#     tuple: The latitude and longitude, or raises a ValueError if the address was not found.
#     """
#     geolocator = Nominatim(user_agent="my-app")  # Create a Nominatim geolocator with a custom user_agent
#     location = geolocator.geocode(address)  # Try to geocode the address
#     if location:
#         return (location.latitude, location.longitude)
#     else:
#         print("Error: Address not found")
#         return False
        

# def get_mapping_lat_long(addresses):
#     """
#     Given a list of addresses, get the latitude and longitude for each address.
    
#     Args:
#     addresses (list): The list of addresses to geocode.
    
#     Returns:
#     dict: A dictionary mapping addresses to (latitude, longitude) tuples.
#     """
#     # Get the coordinates for each address
#     coordinates_list = list(map(get_coordinates, addresses))
#     lat_lon_dict = dict(zip(addresses, coordinates_list))  # Create a dictionary mapping addresses to coordinates
    
#     # Print the results
#     for address, coordinates in lat_lon_dict.items():
#         if coordinates:
#             latitude, longitude = coordinates
#             print(f"Address: {address}")
#             print(f"Latitude: {latitude}")
#             print(f"Longitude: {longitude}")
#             print()
#         else:
#             print(f"Address not found or coordinates not available: {address}")
#             print()
            
#     return lat_lon_dict


# def add_latitude_and_longitude(df, mappings, key_column, lat_column, long_column):
#     """
#     Add latitude and longitude to a dataframe based on an existing column of addresses.
    
#     Args:
#     df (DataFrame): The dataframe to add latitude and longitude to.
#     mappings (dict): A dictionary mapping addresses to (latitude, longitude) tuples.
#     key_column (str): The column of the dataframe that contains the addresses.
#     lat_column (str): The name of the new latitude column.
#     long_column (str): The name of the new longitude column.
#     """
#     # Create dictionaries mapping addresses to latitudes and longitudes
#     latitudes = {k: v[0] for k, v in mappings.items() if v}

#     longitudes = {k: v[1] for k, v in mappings.items() if v}

#     # Map the latitudes and longitudes to the dataframe
#     df[lat_column] = df[key_column].map(latitudes).combine_first(df[lat_column])

#     df[long_column] = df[key_column].map(longitudes).combine_first(df[long_column])
#     return df


####----------------------------------------------------------------------------------------

def clean_df(df, location_key):
    """
    Cleans the input dataframe by filtering based on location key and renaming columns.
    
    Parameters:
    - df: DataFrame to be cleaned.
    - location_key: String representing the location key to filter by.
    
    Returns:
    - df: Cleaned DataFrame.
    """
    
    # Filter the dataframe based on location key and a specific start date.
    df = df.query('location_key == @location_key and date >= "2020-02-10"')
    
    # Convert column names: lowercase and replace spaces with underscores.
    df.columns = [i.lower().replace(' ', '_') for i in df.columns]

    # Set 'date' column as the index after converting to datetime format.
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    
    # Drop the 'location_key' column as it's no longer needed.
    df = df.drop(columns=['location_key'])

    # Display general information and the first few rows of the DataFrame.
    df.info()
    display(df.head())

    return df

def min_max_variance(series):
    """
    Computes the variance of the normalized series.
    
    Parameters:
    - series: Series of data.
    
    Returns:
    - float: Variance of normalized series.
    """
    
    # Return 0 if max and min are the same to avoid division by zero.
    if series.max() == series.min():
        return 0
    
    # Normalize the series.
    series = (series - series.min()) / (series.max() - series.min())
    
    return series.var()


def quick_summary(df, length=15):
    """
    Display a styled summary of the DataFrame.
    
    Parameters:
    - df: DataFrame to summarize.
    - length: Number of rows to display. Default is 15.
    """
    
    display(df.agg(['mean', 'min', 'max', min_max_variance])
            .T
            .sort_values('min_max_variance').head(length)
            .style
            .background_gradient(cmap='coolwarm', subset='min_max_variance', vmin=0, vmax=.1)
            .format(lambda x: "{:,.4f}".format(x).rstrip('0').rstrip('.') if isinstance(x, (float, int)) else x))


def last_first_missing(master_df, long=False):
    """
    Create a DataFrame summarizing the first and last non-missing date for each column.
    
    Parameters:
    - master_df: DataFrame to analyze.
    - long: Boolean to decide if additional statistics like min, max, and mean should be included.
    
    Returns:
    - arranged_df: DataFrame with summary information.
    """
    
    minim = []
    maxim = []
    mmvar = []
    
    # For each column, determine the first and last non-missing date and compute the min_max_variance.
    for col in master_df:
        minim.append(master_df[master_df[col].notna()].index.min())
        maxim.append(master_df[master_df[col].notna()].index.max())
        mmvar.append(min_max_variance(master_df[col]))

    # Construct a DataFrame from the collected data.
    first_last_dates = pd.DataFrame(data={'First_Data_Date': minim, 'Last_Data_Date': maxim, 'percent': 
                                          (master_df.isna().sum() / len(master_df)).values * 100, 'min_max_variance': mmvar}, index=master_df.columns)

    # Convert the date columns to datetime format.
    first_last_dates[['First_Data_Date', 'Last_Data_Date']] = first_last_dates[['First_Data_Date', 'Last_Data_Date']].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d').dt.date)
    
    # If the long flag is set, add additional statistics to the DataFrame.
    if long:
        first_last_dates['min'] =  master_df.min()
        first_last_dates['max'] = master_df.max()
        first_last_dates['mean'] = master_df.mean()

    # Arrange the DataFrame based on the 'percent' column.
    arranged_df = first_last_dates.sort_values(by='percent', ascending=False)
    
    return arranged_df



def spline_testing(df, col, order=5, method='spline'):
    mse_errors, mae_errors, mape_errors = [], [], []

    for i in range(1, order+1, 1):
        
        candidate = df[col][df[col].notna()]

        spline = df[col].interpolate(method=method, order=i)
        spline.loc[candidate.index] = np.nan

        y_true = candidate.iloc[1:-1] 
        y_predicted = spline.interpolate(method=method, order=i).loc[candidate.index].iloc[1:-1]
        
        mse_errors.append(mean_squared_error(y_true, y_predicted))
        mae_errors.append(mean_absolute_error(y_true, y_predicted))
        mape_errors.append(np.mean(np.abs(np.divide(y_true - y_predicted, y_true, out=np.zeros_like(y_predicted, dtype=float), where=y_true!=0))) * 100)

    return pd.DataFrame(data={'mse': mse_errors, 'mae': mae_errors, 'mape': mape_errors}, index=[f"{method} Order " + str(i) for i in range(1, order+1, 1)])



# Time series cross-validation
def train_test_validation(df, col, n_splits=5, order=5, verbose=False, method = 'spline'):
    df_holder_train = []
    df_holder_test = []

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, test_idx in tscv.split(df[col]):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Now, concatenate train and test to get a full dataframe for this split
        # combined_df = pd.concat([train_df, test_df])

        # Call the spline_testing function
        df_holder_train.append(spline_testing(train_df, col, order=order, method= method))
        df_holder_test.append(spline_testing(test_df, col, order=order, method=method))

    if verbose:
        # Now, df_holder will have a list of dataframes with residuals for each split
        for index, df in enumerate(df_holder_train):
            print(f"Train Split {index + 1} Results:")
            display(df)
            print("\n")
            
        # Now, df_holder will have a list of dataframes with residuals for each split
        for index, df in enumerate(df_holder_test):
            print(f"Test Split {index + 1} Results:")
            display(df)
            print("\n")
    
    train_error = reduce(lambda x, y: x + y, df_holder_train)/n_splits
    test_error = reduce(lambda x, y: x + y, df_holder_test)/n_splits
    return train_error, test_error

####----------------------------------------------------------------------------------------

print("Versions used in this notebook:")
print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")
print(f"Seaborn version: {sns.__version__}")
print(f"Matplotlib version: {mpl.__version__}")
print(f"Scipy version: {scipy.__version__}")
print(f"Statsmodels version: {sm.__version__}")
print(f"SKLearn version: {sk.__version__}")
