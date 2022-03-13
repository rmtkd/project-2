# Table of Contents
1. [Installation](https://github.com/rmtkd/project-1/blob/main/README.md#installation)
2. [Project Motivation](https://github.com/rmtkd/project-1/blob/main/README.md#project-motivation)
3. [File Descriptions](https://github.com/rmtkd/project-1/blob/main/README.md#file-descriptions)
4. [Results](https://github.com/rmtkd/project-1/blob/main/README.md#results)
5. [Licensing](https://github.com/rmtkd/project-1/blob/main/README.md#licensing)

# Installation

The project successfully ran in the Udacity workspace IDE with python 3.6.3.

# Project Motivation

In this project we built a disaster response pipeline, which take as input a real message that was sent during disaster events, and outputs the categories of the message. The project contains the following components:
1. ETL Pipeline
2. ML Pipeline
3. Flask Web App

# File Descriptions

- Two jupyter notebooks ([ETL Pipeline Preparation.ipynb](https://github.com/rmtkd/project-2/blob/main/ETL%20Pipeline%20Preparation.ipynb) and [ML Pipeline Preparation.ipynb](https://github.com/rmtkd/project-2/blob/main/ML%20Pipeline%20Preparation.ipynb)) are available, which were used to build the ETL and ML pipelines.
- In the [data](https://github.com/rmtkd/project-2/tree/main/data) folder, you can find [disaster_categories.csv](https://github.com/rmtkd/project-2/blob/main/data/disaster_categories.csv) and [disaster_messages.csv](https://github.com/rmtkd/project-2/blob/main/data/disaster_messages.csv) which contains the data used to train the ML pipeline, after data preparation using the [process_data.py](https://github.com/rmtkd/project-2/blob/main/data/process_data.py) file. The process_data.py takes the two csvs as input, and results a database [DisasterResponse.db](https://github.com/rmtkd/project-2/blob/main/data/DisasterResponse.db) which contains the processed data.
- In the [models](https://github.com/rmtkd/project-2/tree/main/models) folder, you can find the [train_classifier.py](https://github.com/rmtkd/project-2/blob/main/models/train_classifier.py), which trains the resulting data from DisasterResponse.db and outputs a pickle file classifier.pkl, which is not available in the repository due the file size.
- In the [app](https://github.com/rmtkd/project-2/tree/main/app) folder, you can find tamplates folder, which contains the html templates used for the web app, and the [run.py](https://github.com/rmtkd/project-2/blob/main/app/run.py). which is the main file to run the project and host the web app.

# Results

The ML pipeline had a better result using multiple outputs Random Forest, with 200 estimators and an accuracy of 94,6%.
The run.py web app runs in the terminal without errors, and the main page includes three visualizations of the data, including:
- Genre distribution
- Categories distribution
- Categories distribution with genre breakdown

# Licensing

Credits to Udacity by providing the templates for the project.
