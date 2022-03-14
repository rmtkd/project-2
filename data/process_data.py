import sys
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ 
    Loads all the datasets needed.
    
    Args:
        messages_filepath (string): name of messages dataset
        categories_filepath (string): name of categories dataset
    
    Returns:
        df (dataframe): dataframe with messages and categories datasets merged
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='inner')
    return df



def clean_data(df):
    """ 
    Cleans the data from the loaded datasets. 
    
    Args:
        df (dataframe): merged dataframe with messages and categories
        
    Returns:
        df (dataframe): dataframe with cleaned dataframe
    """
    categories = df['categories'].str.split(";", expand=True)
    row = categories.loc[0, :]
    category_colnames = row.apply(lambda x: x[0:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])
        categories[column] = np.where(categories[column]>1, 1, categories[column])
    df = df.drop('categories', 1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates(subset=['message', 'original', 'genre'])
    return df


def save_data(df, database_filename):
    """ 
    Stores the input dataframe to a sql database. 
    
    Args:
        df (dataframe): cleaned dataframe
        database_filename (string): name of database to store the dataframe
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_categories', engine, index=False, if_exists='replace')


def main():
    """ 
    Runs the ETL pipeline. 
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()