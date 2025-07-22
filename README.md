# Airline Customer Satisfaction Project 

## Repository Outline
1. `.streamlit/`
   - Configuration folder for Streamlit (usually contains `config.toml`).
2. `deployment/`
   - 2.1 `app.py` – Main Streamlit app controller for navigation and launch.
   - 2.2 `eda.py` – Handles Exploratory Data Analysis view.
   - 2.3 `mainapp.py` – Displays project background and model explanation.
   - 2.4 `prediction.py` – Manages input form and shows prediction result.
   - 2.5 `sidebar.py` – Sidebar navigation and layout.
   - 2.6 `model_xgb.pkl` – Serialized XGBoost model used in prediction.
   - 2.7 `model.joblib` – Alternative serialized model version.
   - 2.8 `train.csv` – Cleaned training data used for model building.
   - 2.9 `plots.py` – Plotting function based on matplotlib.
3. `venv/`
   - Virtual environment folder (excluded from version control).
4. `.gitignore` – Specifies files and folders to be ignored by Git.
5. `archive.ipynb` – Alternate hyperparameter boosting for SVM.
6. `description.md` – Main documentation for the repository.
7. `mlanalyzer.py` – Script for model training, tuning, or analysis.
8. `model_svm.pkl` – Serialized SVM model for backup.
9. `model_xgb.pkl` – XGBoost model in root.
10. `P1G5_Faishal-Kemal.ipynb` – Initial exploratory and modeling notebook.
11. `P1G6_Faishal-Kemal.ipynb` – Continued or extended version of analysis.
12. `P1M2_Faishal-Kemal_conceptual.txt` – Conceptual plan for the project.
13. `P1M2_Faishal-Kemal_Inference.ipynb` – Notebook for final inference workflow.
14. `P1M2_Faishal-Kemal.ipynb` – Final version of notebook with all steps.
15. `requirements.txt` – Required packages to run the project.
16. `test.csv` – Dataset used for testing/evaluation.
17. `train.csv` – Original training data.
18. `url.txt` – Contains external links, for data source and deployment.

## Problem Background
Airline market is a competitive market. In 2025, the global airline industry is expected to generate a total revenue of $979 billion, according to IATA.

Customer satistfaction can affect the rate of income, as the higher the satistfaction, there is higher chance that the customer will return, or even recommend someone else.

As an data scientist, the aim for the project is to create a model that can predict the satistfaction of the customer. that the company can use to see on which section they can improve on.

## Project Output
The main output of the project is a trained model, that user can use either directly using `inference.ipynb` or by accessing the deployed model on either *huggingface* or *streamlit*.

Several visualization based on the `Exploratory Data Analysis` can be read on the main notebook, or on the deployed site.

## Data
dataset used is `Airline Passenger Satisfaction` taken from Kaggle.

for original dataset can be seen in this [link](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data).

| Column Name                         | Data Type | Description                                                                 |
|------------------------------------|-----------|-----------------------------------------------------------------------------|
| id                                 | INT64     | Unique identifier for each passenger                                       |
| Gender                             | STRING    | Gender of the passenger (Female, Male)                                     |
| Customer Type                      | STRING    | Loyalty classification (Loyal customer, Disloyal customer)                 |
| Age                                | INT64     | Age of the passenger                                                       |
| Type of Travel                     | STRING    | Purpose of travel (Personal Travel, Business Travel)                       |
| Class                              | STRING    | Class of travel (Business, Eco, Eco Plus)                                  |
| Flight Distance                    | INT64     | Distance of the flight in miles                                            |
| Inflight wifi service              | INT64     | Satisfaction with inflight wifi (0: Not Applicable, 1–5 scale)             |
| Departure/Arrival time convenient  | INT64     | Satisfaction with departure/arrival times (1–5 scale)                      |
| Ease of Online booking             | INT64     | Satisfaction with online booking (1–5 scale)                               |
| Gate location                      | INT64     | Satisfaction with gate location (1–5 scale)                                |
| Food and drink                     | INT64     | Satisfaction with food and drink (1–5 scale)                               |
| Online boarding                    | INT64     | Satisfaction with online boarding process (1–5 scale)                      |
| Seat comfort                       | INT64     | Satisfaction with seat comfort (1–5 scale)                                 |
| Inflight entertainment             | INT64     | Satisfaction with inflight entertainment (1–5 scale)                       |
| On-board service                   | INT64     | Satisfaction with overall onboard service (1–5 scale)                      |
| Leg room service                   | INT64     | Satisfaction with leg room (1–5 scale)                                     |
| Baggage handling                   | INT64     | Satisfaction with baggage handling (1–5 scale)                             |
| Checkin service                    | INT64     | Satisfaction with check-in service (1–5 scale)                             |
| Inflight service                   | INT64     | Satisfaction with inflight service (1–5 scale)                             |
| Cleanliness                        | INT64     | Satisfaction with cleanliness of the flight (1–5 scale)                    |
| Departure Delay in Minutes         | INT64     | Delay at departure in minutes                                              |
| Arrival Delay in Minutes           | FLOAT64   | Delay at arrival in minutes                                                |
| satisfaction                       | STRING    | Overall satisfaction (Satisfaction, Neutral or Dissatisfaction)           |

Data already splitted into train and test, with 80-20 rate. <br>
the original dataset has 103.904 data on train and 25.976 on test. <br>
With 22 feature, and one target.

## Method
- K-Nearest Neighbors (KNN) <br>
This model predicts the label of new data by looking at the closest data points and choosing the majority class among them.

- Support Vector Machine (SVM) <Br>
SVM works by finding the best boundary that separates classes with the largest margin, helping the model generalize well.

- Decision Tree <br>
A model that splits the data into branches based on feature values and follows the path to reach a final prediction.

- Random Forest <br>
An ensemble of decision trees that takes the majority vote from multiple trees, making it more robust and less prone to overfitting.

- XGBoost <BR>
A boosting model that builds trees one by one, each time correcting errors from the previous tree, making it very powerful and accurate.

## Stacks
This project will be carried out using Python for data cleaning, data exploration analysis, modeling, predicting, and visualization. 

Key libraries that will be used here :
- `pandas`, `numpy` - For dataset storage and manipulation
- `matplotlib`, `seaborn` - For visualization
- `scipy` - For analysis
- `pickle` - For export and import model
- `scikit-learn` - For imputing, modeling. and evaluating
- `statsmodels` - For assumption test
- `xgboost` - boosting algorithm based on tree based algorithm.
- `mlanalyser` - My own script consist of function for cleaner notebook

## Reference
Online deployment has been done in 2 different website.
- HuggingFace - [AirlineSatisfactionProject](https://huggingface.co/spaces/vikesaki/AirlineSatisfactionProject)
- StreamLit - [airlinecustomersatisfactionproject](https://airlinecustomersatisfactionproject.streamlit.app)

Notes - HuggingFace is buggy, as the paging system didnt actually clear the previous page, and the screen width setting make the page 'jittery'
for better experience, i recommend to use the streamlit ones.

---

**Referensi tambahan:**
- [A Guide to Principal Component Analysis (PCA) for Machine Learning](https://www.keboola.com/blog/pca-machine-learning#:~:text=PCA%20assumes%20a%20linear%20relationship,methods%20such%20as%20log%20transforms.)
- [Principal Components Analysis for Ordinal Data](https://www.researchgate.net/publication/381280495_Principal_Components_Analysis_for_Ordinal_Data_using_R)
- [PCA Usage](https://www.ibm.com/docs/en/ias?topic=pca-usage)