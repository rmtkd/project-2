import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import plotly.graph_objs as go

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Data for graph 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Data for graph 2
    category_counts = df.iloc[:, 4:].melt().groupby('variable').value.sum().sort_values(ascending=False)
    category_names = list(category_counts.index)
    
    # Data for graph 3
    categories_df = df.iloc[:, 3:].groupby('genre').sum().transpose()
    categories_df.sort_values('direct', ascending=False, inplace=True)
    genre_list = list(categories_df.columns)
    x_val = categories_df.index.tolist()
    graph_three = []
    for genre in genre_list:
      y_val = categories_df.loc[:,genre].tolist()
      graph_three.append(
          Bar(
              x = x_val,
              y = y_val,
              name = genre
          )
      )
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': graph_three,
            
            'layout': {
                'title': 'Distribution of Message Categories per genre',
                'barmode': 'stack',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """ 
    Runs the web app.
    """
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()