# Customer-churn-
# Telco Customer Churn Prediction

This project focuses on building a machine learning model to predict customer churn in a telecommunications dataset. Customer churn, the phenomenon of customers discontinuing their service, is a significant challenge for businesses. By accurately identifying customers at risk of churning, companies can implement proactive retention strategies, which are often more cost-effective than acquiring new customers.

## Project Overview

The goal of this project is to develop a robust predictive model that can identify potential churners. The Jupyter Notebook (`telco churn.ipynb`) outlines the complete machine learning pipeline, from data exploration to model evaluation.

## Dataset

The dataset used in this project is `WA_Fn-UseC_-Telco-Customer-Churn.csv`. It contains information about a fictional telecommunications company's customers, including their services, account information, and whether they churned or not.

## Key Stages of the Project

The `telco churn.ipynb` notebook covers the following main stages:

### 1. Data Loading and Initial Exploration (EDA)

* **Loading Data:** The project begins by loading the customer churn dataset into a Pandas DataFrame.
* **Initial Inspection:** Basic data checks are performed, including:
    * Checking the shape of the dataset (number of rows and columns).
    * Inspecting data types of each column using `.info()`.
    * Generating descriptive statistics for numerical features using `.describe()`.
* **Univariate & Bivariate Analysis:** Visualizations (e.g., count plots, bar charts, histograms, box plots) are used to understand:
    * The distribution of individual features.
    * The relationship between various features and the `Churn` target variable (e.g., how `Contract` type, `InternetService`, `PaymentMethod`, `gender`, `tenure` influence churn).

### 2. Data Preprocessing

* **Handling Missing Values:** Specifically addresses missing values, likely in the `TotalCharges` column, by converting them to a numerical format and handling `NaN` values (e.g., imputation or removal of rows).
* **Feature Engineering:** Potential creation of new features to capture more complex relationships within the data (e.g., ratios or tenure groups).
* **Categorical Encoding:** Converts non-numerical categorical features into a numerical format suitable for machine learning algorithms. This typically involves:
    * **One-Hot Encoding:** For nominal categorical variables (e.g., `gender`, `Partner`, `InternetService`, `PaymentMethod`).
    * **Label Encoding:** For ordinal categorical variables (if any, though less common in this dataset for direct ordering).
* **Feature Scaling:** Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) are scaled (e.g., using `StandardScaler`) to normalize their ranges, preventing features with larger values from disproportionately influencing the model.

### 3. Model Building and Evaluation

* **Data Splitting:** The preprocessed data is split into training and testing sets to evaluate the model's performance on unseen data.

* **Training and Prediction:** Selected models are trained on the training data and then used to make predictions on the test set.
* **Model Evaluation Metrics:** The performance of the models is rigorously assessed using a variety of metrics:
    * **Accuracy:** Overall correctness of predictions.
    * **Precision, Recall, F1-Score:** Metrics providing a more nuanced view of classification performance, especially important for imbalanced datasets (churn is often imbalanced).
    * **Confusion Matrix:** Visual representation of correct and incorrect predictions for each class.
    * **ROC AUC Curve:** Measures the classifier's ability to distinguish between churners and non-churners across various threshold settings.
    * **Classification Report:** Provides a summary of precision, recall, and F1-score for each class.

### 4. Insights and Conclusion

* The notebook concludes with a summary of the findings, including the performance of the chosen model(s).
* Identification of key features that significantly influence customer churn (e.g., contract type, internet service, tenure, monthly charges).
* Potential business recommendations derived from the analysis to help the telecommunications company reduce customer churn.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn


## How to Run the Notebook

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```
3.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook "telco churn.ipynb"
    ```
4.  Run all cells in the notebook to reproduce the analysis and model training.
