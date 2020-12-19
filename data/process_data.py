import sys
import pandas as pd
# import numpy as np
# import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function load data from two csv file and then merge them

    INPUT
    messages_filepath - the file path of messages file
    categories_filepath - the file path of categories file


    OUTPUT
    df - A dataframe of merged messages and categories
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on=['id'])
    return df

def clean_data(df):
    """
    This function clean the Dataframe using following steps to produce the new df

    INPUT
    df - A dataframe of messages and categories need to be cleaned

    OUTPUT
    df - A cleaned dataframe of messages and categories
    """

    # split `categories` and create a dataframe of the individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.head(1)

    # use this row to extract a list of new column names for categories.
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)

    # drop the duplicates.
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    This function save the Dataframe df into the database file

    INPUT
    df - A dataframe of messages and categories
    database_filename - The file name of the database

    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_disaster', engine, index=False)

def main():
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