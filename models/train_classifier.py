# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
import pickle
import nltk
# nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
    This function is to load the database from the given filepath and process them as X, Y and category_names

    INPUT:
    database_filepath - Database filepath

    OUTPUT:
    X - Get Message data in dataframe
    Y - Extract Category data
    category_names - List of categories name
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("Messages", con=engine)
    # df = df.head(10)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    This function split text into words and return the root form of the words

    INPUT:
    text - Input message

    OUTPUT:
    clean_tokens - A list of the root form of words
    """
    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    This function is to build a pipeline, process the message, apply classifier to build a classification model

    INPUT:
    N/A

    OUTPUT:
    cv - The classification model
    """
    model = AdaBoostClassifier()

    pipeline_optimize = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(model))
    ])

    parm1 = 'tfidf__use_idf'
    parm2 = 'clf__estimator__n_estimators'
    parm3 = 'clf__estimator__learning_rate'

    parameters = {
        parm1: (True, False),
        parm2: [50, 100],
        parm3: [1, 2]
    }
    cv = GridSearchCV(pipeline_optimize, param_grid = parameters)

    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Function:
    This function to use the classification model to predict the category, and generate report

    INPUT:
    Model - Classification model
    X_test - messages
    Y_test - categories

    OUTPUT:
    Print the classification report
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print('Category {}: {}'.format(i + 1, col))
        print(classification_report(Y_test[col], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    This function is to save the model as a pickle file

    INPUT:
    model - The classification model generated
    model_filepath - the path of pickle file to be saved
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
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
        evaluate_model(model, X_test, Y_test)

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