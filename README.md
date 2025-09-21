# Credit Loan Default Prediction

## Project Overview
This project focuses on predicting loan defaults using machine learning models based on the Lending Club dataset. The analysis is structured into three main parts:

1. **Data Cleaning and Preprocessing**: Handling missing values, feature selection, and data preparation
2. **Closed Loans Analysis**: Building predictive models for loans that have completed their term
3. **Open Loans Analysis**: Applying trained models to predict defaults on currently active loans

## Dataset
The dataset (`Loan_Lending_Club.csv`) contains comprehensive information about loans, including:
- Loan characteristics (amount, interest rate, term, purpose)
- Borrower information (income, employment, credit history)
- Payment history
- Loan status and outcomes

## Project Structure
- `RUN_FIRST_data_cleaning.ipynb`: Initial data cleaning, feature selection, and preparation
- `RUN_SECOND_closed_loans.ipynb`: Analysis and modeling of closed loans
- `RUN_THIRD_open_loans.ipynb`: Application of models to predict defaults on open loans
- `Loan_Lending_Club.csv`: Raw dataset (large file, ~1.5GB)
- `Assignment 1- BusinessCase - Supervised Classification.pdf`: Project requirements and business context
- `AI2_Assignment_1_Darren.pdf`: Project report and findings

## Workflow

### 1. Data Cleaning and Preprocessing
- Loading and exploring the dataset
- Handling missing values
- Feature selection based on correlation analysis and domain knowledge
- Removing redundant and irrelevant features
- Creating separate datasets for closed and open loans
- Exporting processed data as parquet files for subsequent analysis

### 2. Closed Loans Analysis
- Loading preprocessed closed loans data from parquet files
- Feature engineering and selection through multiple methods:
  - Preventing data leakage by removing payment status information
  - Using Random Forest for initial feature importance assessment
  - Selecting features above mean importance threshold (21 key features identified)
  - Interest rate, loan term, and grade emerged as the most important predictors
- Training and evaluating multiple machine learning models:
  - Logistic Regression: Baseline model with interpretable coefficients
  - Random Forest: Ensemble method with feature importance capabilities
  - XGBoost: Gradient boosting algorithm with high predictive performance
- Hyperparameter tuning using RandomizedSearchCV with stratified k-fold cross-validation
- Comprehensive model evaluation:
  - Random Forest achieved the best performance with AUC of 0.712
  - Accuracy of 0.642 across all models
  - Precision around 0.32 (identifying true defaults)
  - Recall of 0.67 (capturing 67% of actual defaults)
  - F1-score of 0.43 (harmonic mean of precision and recall)
- Detailed confusion matrix analysis showing trade-offs between false positives and false negatives

### 3. Open Loans Analysis
- Loading preprocessed open loans data from parquet files
- Applying the best model (Random Forest) from closed loans analysis
- Predicting default probabilities for currently active loans
- Threshold optimization:
  - Initial threshold from closed loans (0.51) applied
  - Optimized threshold for open loans determined to be 0.47
  - ROC curve analysis to maximize true positive rate while minimizing false positives
- Implementing three distinct investment strategies based on risk tolerance:
  - Conservative strategy (grades A-B): Lower default risk, lower returns
  - Moderate strategy (grades C-D): Balanced risk-return profile
  - Aggressive strategy (grades E-G): Higher default risk, higher potential returns
- Each strategy received custom-optimized thresholds through ROC curve analysis
- Comprehensive business impact assessment:
  - Quantified missed revenue ($2.92B) from rejecting profitable loans
  - Calculated default losses ($87.4M) from incorrectly approved loans
  - Estimated expected interest earned ($1.27B)
  - Detailed financial impact analysis for each investment strategy

## Key Features
The project focuses on identifying the most relevant features for predicting loan defaults, including:
- Interest rate
- Loan term
- Credit grade
- Debt-to-income ratio
- Employment length
- Annual income
- Loan purpose

## Models
Several classification models are implemented and compared:
- Logistic Regression: Baseline model with interpretable coefficients
- Random Forest: Ensemble method with feature importance capabilities
- XGBoost: Gradient boosting algorithm with high predictive performance

## Results
The models are evaluated based on:
- Classification accuracy
- Precision and recall
- F1-score
- ROC-AUC
- Confusion matrix analysis
- Business impact assessment

## Installation and Usage

### Requirements
All required libraries are listed in `requirements.txt`. Install them using:
```
pip install -r requirements.txt
```

### Running the Analysis
Run the notebooks in the following order:
1. `RUN_FIRST_data_cleaning.ipynb`
2. `RUN_SECOND_closed_loans.ipynb`
3. `RUN_THIRD_open_loans.ipynb`

Note: The raw dataset is large (~1.5GB), so ensure you have sufficient memory available.

## Business Application
This project demonstrates how machine learning can be applied to:
- Assess credit risk more accurately
- Identify potential defaults before they occur
- Optimize lending strategies based on risk profiles
- Improve decision-making in loan approval processes

## Future Improvements
Potential enhancements for the project:
- Implementing more advanced models (neural networks, stacked ensembles)
- Incorporating additional external data sources
- Developing a real-time prediction system
- Creating a web interface for interactive analysis
- Addressing class imbalance with advanced techniques
- Exploring explainable AI methods for model interpretation

## License
This project is for educational purposes as part of the Artificial Intelligence course at ESADE Business School.
