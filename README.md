# Machine Learning Model for Disease Prediction with synthetic data

**Objective: To build predictive models that classify disease risk using genomic data (SNP/genotype information).**

**Approach:**

Preprocessing genotype data into machine-learning–friendly formats.

Splitting data into training and testing sets.

Training classification models (e.g., Logistic Regression, Random Forest).

Evaluating model accuracy and feature importance.

Generating predictions and visualizations.

**Repository Structure**

prepare_genotype_data*.py → Scripts for preprocessing genotype datasets.

train_model1.py, trainmodel.py → Scripts for training ML models.

logistic_regression_model.pkl, random_forest_model.pkl → Saved trained models.

visualization.py → Visualization of feature importance and results.

feature_importance.png → Visual representation of important genetic features.

X_train_preprocessed.csv, X_test_preprocessed.csv → Preprocessed input data.

y_train.csv, y_test.csv → Labels for training and testing.

predictions.csv → Model predictions on test data.

**Methods & Tools**

Languages/Frameworks: Python, Pandas, Scikit-learn, Matplotlib.

Models Used: Logistic Regression, Random Forest.

Genomic Data: Encoded SNP/genotype datasets.

**Key Insights**

Logistic Regression and Random Forest models can predict disease risk from genomic data.

Feature importance analysis helps identify SNPs most strongly associated with disease traits.

Preprocessing steps (e.g., encoding, cleaning) are critical for accurate predictions.

**Future Directions**

Extend to larger-scale GWAS datasets.

Apply deep learning models for higher prediction accuracy.

Integrate pharmacogenomics knowledge bases (PharmGKB, ClinVar) for drug–gene interactions.
