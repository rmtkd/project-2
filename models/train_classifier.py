import sys
import sqlalchemy
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import re
import numpy as np
import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """ 
    Loads all the data stored in SQL database, and returns X, y and categories names, which are 
    used further on the supervised training
    
    Args:
        database_filepath (string): name of processed data sql database
    
    Returns:
        X (list): array with messages
        y (list): array with binary of each category
        category_names (list): list of category names
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM disaster_categories", engine)
    X = df.message.values
    y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns
    return X, y, category_names


def tokenize(text):
    """ 
    Tokenize, lemmatizes and cleans the input text.
    
    Args:
        text (string): inputted text to be cleaned
    
    Returns:
        clean_tokens (list): array with cleaned text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ 
    ML pipeline with Grid Search for parameters tunning.
    
    Returns:
        cv (model): Grid Search with text processing machine learning pipeline
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__criterion': ['gini', 'entropy']
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ 
    Prints the f1 score, precision and recall for the test set.
    
    Args:
        model (model): name of messages dataset
        X_test (list): list with test values
        Y_test (list): list with test values
        category_names (list): categories names
    """
    y_pred = model.predict(X_test)
    accuracy = (y_pred == Y_test).mean()
    for i in range(len(category_names)):
        print(category_names[i], classification_report([row[i] for row in y_test], [row[i] for row in y_pred]))


def save_model(model, model_filepath):
    """ 
    Saves model to pickle file.
    
    Args:
        model (model): ML model
        model_filepath (string): name of output pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ 
    Runs the ML pipeline
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()