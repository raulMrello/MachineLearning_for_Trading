# MachineLearning for Trading

Project in which I study the implementation of an intelligent system capable of making investments in the FOREX market.

The basic idea of the project can be divided into several phases:

- Design of a price forecasting system
- Design of an intelligent agent that performs trading actions on a given forex pair (eg eur-usd)
- Design of an interface with the electronic trading platform to obtain the data-feed in real time and the status of the account / operations.

As a development guide, I am using PYTHON language and these most representative libraries:

- scikit-learn: several ML utilities (cross-validation, GridSearch, LinearRegressores, ...)
- tensorflow: low level deep learning library
- keras: tensorflow front-end (application level)
- featurestools: feature engineering
- deap: implementation of genetic algorithms

  
## Changelog


-------------------------------------------------- --------------------------------------------
##### 08.01.2019 -> commit: "Update predictions with H4 histdata"
- [x] Update generation of predictions in Price prediction notebook with H4 data
- [x] Add images, documents, csv results and common documentation
-------------------------------------------------- --------------------------------------------
##### 28.12.2018 -> commit: "Update notebook with generated predictions"
- [x] Update generation of predictions in Price prediction notebook
- [x] Add images, documents, csv results and common documentation
-------------------------------------------------- --------------------------------------------
##### 20.12.2018 -> commit: "Curated notebook for price prediction"
- [x] Refactor content in Price prediction notebook
- [] Next stuff: integrate Gym into the A3C-LSTM notebook and implement data 'pullers' from the dataframe df_a3c_in. Once verified, implement the agent.
-------------------------------------------------- --------------------------------------------
##### 12.12.2018 -> commit: "Included statistics in TradeSim-v0"
- [x] Included statistics, rewards and correct errors. Current version v0.0.5
- [x] Updated notebooks.
- [] Next stuff: integrate Gym into the A3C-LSTM notebook and implement data 'pullers' from the dataframe df_a3c_in. Once verified, implement the agent.
-------------------------------------------------- --------------------------------------------
##### 11.12.2018 -> commit: "Implemented Gym TradeSim-v0"
- [x] Completed first version (v0.0.1) of the environment for Gym OpenAI of the trading simulator. I have to define rewards correctly and create a new version (v0.0.2)
- [x] Added notebook with gym test env created.
-------------------------------------------------- --------------------------------------------
##### 05.12.2018 -> commit: "Designing gym-environment"
- [x] Starts development of Gym-env that serves as a reference to evaluate the A3C agent.
- [x] Clone several repos in ./gym

-------------------------------------------------- --------------------------------------------
##### 04.12.2018 -> commit: "Starting design A3C-LSTM"
- [x] Discard post-predictor filter since it does not introduce any improvement.
- [x] Created notebook to generate prediction dataframe and generate CSV for the A3C Agent
- [x] Started  A3C-LSTM agent deployment notebook
- [x] Created RAR files for heavy CSV files and extract the CSV from the backup copy


