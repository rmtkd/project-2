# Table of Contents
1. [Installation](https://github.com/rmtkd/project-1/blob/main/README.md#installation)
2. [Project Motivation](https://github.com/rmtkd/project-1/blob/main/README.md#project-motivation)
3. [File Descriptions](https://github.com/rmtkd/project-1/blob/main/README.md#file-descriptions)
4. [Results](https://github.com/rmtkd/project-1/blob/main/README.md#results)
5. [Licensing](https://github.com/rmtkd/project-1/blob/main/README.md#licensing)

# Installation

The project successfully ran in the Udacity workspace IDE with python 3.6.3.

# Project Motivation

After a disaster, millions of messages are sent direct or via social media, and organizations need to filter which messages are really important and relevant, so in this project we built a disaster response pipeline, which take as input a real message that was sent during disaster events, and outputs the categories of the message, so different organizations can take care of different parts of the problem.
The project contains the following components:
1. ETL Pipeline
2. ML Pipeline
3. Flask Web App

# File Descriptions

- [app](https://github.com/rmtkd/project-2/tree/main/app)
  - [run.py](https://github.com/rmtkd/project-2/blob/main/app/run.py) _Flask file that runs the app_
  - [templates](https://github.com/rmtkd/project-2/tree/main/app/templates)
    - [go.html](https://github.com/rmtkd/project-2/blob/main/app/templates/go.html) _main page of web app_
    - [master.html](https://github.com/rmtkd/project-2/blob/main/app/templates/master.html) _classification result page of web app_
- [data](https://github.com/rmtkd/project-2/tree/main/data)
  - [disaster_categories.csv](https://github.com/rmtkd/project-2/blob/main/data/disaster_categories.csv) _data to process_
  - [disaster_messages.csv](https://github.com/rmtkd/project-2/blob/main/data/disaster_messages.csv) _data to process_
  - [process_data.py](https://github.com/rmtkd/project-2/blob/main/data/process_data.py) _python script with ETL pipeline_
  - [DisasterResponse.db](https://github.com/rmtkd/project-2/blob/main/data/DisasterResponse.db) _database with cleaned data_
- [models](https://github.com/rmtkd/project-2/tree/main/models)
  - [train_classifier.py](https://github.com/rmtkd/project-2/blob/main/models/train_classifier.py) _python script with the ML pipeline_
- [README.md](https://github.com/rmtkd/project-2/blob/main/README.md)
- [ETL Pipeline Preparation.ipynb](https://github.com/rmtkd/project-2/blob/main/ETL%20Pipeline%20Preparation.ipynb) _ETL pipeline preparation_
- [ML Pipeline Preparation.ipynb](https://github.com/rmtkd/project-2/blob/main/ML%20Pipeline%20Preparation.ipynb) _ML pipeline preparation_

To run the python scripts and the web app in the terminal:
- To run ETL pipeline that cleans data and stores in database:
    ```
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```
- To run ML pipeline that trains classifier and saves:
    ```
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    ```
- To run the web app:
  - Go to `app` directory: `cd app`
  - Run your web app: `python run.py`
  - Click the `PREVIEW` button to open the homepage

# Results

The ML pipeline had a better result using multiple outputs Random Forest, with 200 estimators and an accuracy of 94,6%.
The run.py web app runs in the terminal without errors, and the main page includes three visualizations of the data, including:
- Genre distribution
- Categories distribution
- Categories distribution with genre breakdown

# Licensing

Credits to Udacity by providing the templates for the project.
