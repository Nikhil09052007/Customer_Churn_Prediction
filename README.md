
# Customer Churn Prediction

A machine learning project that predicts whether a customer will churn (stop using a service) based on a small set of customer attributes. The project covers data exploration, cleaning, feature engineering, model training and comparison, and deployment as a Streamlit web app.

## Project Goal

Given basic customer information, predict whether the customer is likely to churn (`Yes`) or not (`No`).

## Dataset

The dataset is a customer churn CSV file expected at `data/customer_churn_data.csv`. It contains the following columns used in this project:

- `Age` - customer age
- `Gender` - Male or Female
- `Tenure` - number of months the customer has used the service
- `MonthlyCharges` - monthly billing amount
- `ContractType` - type of contract the customer is on
- `InternetService` - type of internet service
- `Churn` - target column, Yes or No

Note: the dataset file itself is not included in this repository. It needs to be placed in a `data/` folder before running the notebook from the beginning.

## Project Structure

```
.
├── main.ipynb          # Full workflow: EDA, cleaning, feature engineering, model training
├── app.py               # Streamlit app for serving predictions
├── model.pkl             # Trained classifier, exported with joblib
├── scaler.pkl             # Fitted StandardScaler, exported with joblib
├── eda_report.html         # Automated EDA report generated with ydata-profiling
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── .gitignore
```

## Workflow

### 1. Data Loading

The dataset is loaded into a pandas DataFrame using `pd.read_csv`.

### 2. Exploratory Data Analysis (EDA)

- An automated profiling report is generated using `ydata-profiling` and saved as `eda_report.html`.
- Basic checks are performed: `df.info()`, `df.shape`, `df.describe()`, missing value counts, and duplicate row checks.

### 3. Data Cleaning

- The `InternetService` column had missing values; these are filled in (imputed) rather than dropped.
- Duplicate rows are checked for.

### 4. Analysis and Visualization

- Correlation between numerical columns is computed with `df.corr()`.
- Churn counts are visualized with a pie chart.
- Average monthly charges and tenure are compared across churned vs. non-churned customers, and by contract type.
- A histogram is generated for `MonthlyCharges` and `Tenure`.
- Saved charts are written to a `charts/` folder, created automatically if it does not already exist.

### 5. Feature Engineering

- The feature set (`X`) is built from four columns: `Age`, `Gender`, `Tenure`, `MonthlyCharges`.
- `Gender` is converted to a numeric value: `Female = 1`, `Male = 0`.
- The target column `Churn` is kept as `Yes` / `No` text labels; scikit-learn classifiers used here accept string class labels directly, so no separate numeric encoding step is required for the target.

### 6. Train/Test Split

The data is split into training and test sets using `train_test_split`, with 20% of the data held out for testing.

### 7. Scaling

A `StandardScaler` is fit on the training features only, then used to transform both the training and test features. The fitted scaler is exported as `scaler.pkl` so the same scaling is applied later at prediction time in the app.

### 8. Model Training and Selection

Three classifiers are trained and compared on accuracy:

- Logistic Regression
- K-Nearest Neighbors, tuned with `GridSearchCV` over `n_neighbors` and `weights`
- Support Vector Classifier (SVC), tuned with `GridSearchCV` over `C` and `kernel`

The best-performing model from the grid search is exported as `model.pkl` using joblib.

### 9. Deployment

`app.py` is a Streamlit application that:

- Loads `scaler.pkl` and `model.pkl`
- Collects `Age`, `Tenure`, `MonthlyCharges`, and `Gender` from the user through input widgets
- Scales the input using the saved scaler
- Predicts churn with the saved model and displays the result as `Yes` or `No`

## Setup and Usage

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

   Note: `requirements.txt` lists the libraries used for data analysis and modeling. Running `app.py` also requires `streamlit` and `joblib`, and re-running the EDA cell in the notebook requires `ydata-profiling`. Install any of these separately if they are not already present in your environment.
2. To explore or re-run the analysis and training steps, open `main.ipynb` in Jupyter. This requires the dataset to be present at `data/customer_churn_data.csv`.
3. To run the prediction app:

   ```
   streamlit run app.py
   ```

   This uses the already-trained `model.pkl` and `scaler.pkl`, so the dataset is not required just to use the app.

## Files Produced by the Notebook

- `model.pkl` - the trained classifier
- `scaler.pkl` - the fitted scaler used to transform input features before prediction
- `eda_report.html` - automated profiling report of the raw dataset
- `charts/` - saved chart images (pie chart of churn counts, histogram of tenure)

## Notes and Limitations

- The model is trained on a small, fixed feature set (`Age`, `Gender`, `Tenure`, `MonthlyCharges`). Other available columns, such as `ContractType` and `InternetService`, are explored during EDA but not used as model inputs.
- Model performance is evaluated using accuracy only. Depending on how balanced the `Churn` classes are, other metrics such as precision, recall, or F1-score may give a more complete picture and are worth adding if this project is extended.
- The dataset file is not included in this repository and must be supplied separately to reproduce the full notebook workflow from scratch.

