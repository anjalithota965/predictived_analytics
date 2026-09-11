# ============================================================
# SUPPLY CHAIN PREDICTIVE ANALYTICS - COMPLETE ANALYSIS SCRIPT
# Topic:
# Evaluating the Adoption of Predictive Analytics
# in Supply Chain Risk Management
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

output_folder = "supply_chain_analysis_results"

os.makedirs(
    output_folder,
    exist_ok=True
)

print("\nOutput folder:", output_folder)


# ============================================================
# 3. LOAD DATASET
# ============================================================

file_path = "/Users/sandeepreddy/Downloads/supply_chain_resilience_dataset.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 4. VIEW DATASET
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

missing_values = df.isnull().sum()

missing_values_df = pd.DataFrame({
    "Column": missing_values.index,
    "Missing_Values": missing_values.values
})

missing_values_df.to_csv(
    os.path.join(
        output_folder,
        "missing_values.csv"
    ),
    index=False
)

print("\nMissing Values:")
print(missing_values_df)


# ============================================================
# 6. CHECK AND REMOVE DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows:", duplicate_count)

df = df.drop_duplicates().copy()

print(
    "Rows after removing duplicates:",
    len(df)
)


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

if "Disruption_Type" in df.columns:

    df["Disruption_Type"] = (
        df["Disruption_Type"]
        .fillna("No Disruption")
    )


if "Disruption_Severity" in df.columns:

    df["Disruption_Severity"] = (
        df["Disruption_Severity"]
        .fillna("None")
    )


# Fill numeric missing values with median

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    if df[column].isnull().sum() > 0:

        df[column] = (
            df[column]
            .fillna(
                df[column].median()
            )
        )


# Fill remaining categorical missing values with mode

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        mode_value = df[column].mode()

        if not mode_value.empty:

            df[column] = (
                df[column]
                .fillna(
                    mode_value[0]
                )
            )


# ============================================================
# 8. CONVERT DATE COLUMNS
# ============================================================

date_columns = [
    "Order_Date",
    "Dispatch_Date",
    "Delivery_Date"
]

for column in date_columns:

    if column in df.columns:

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )


# ============================================================
# 9. SAVE CLEANED DATASET
# ============================================================

cleaned_file = os.path.join(
    output_folder,
    "supply_chain_cleaned.csv"
)

df.to_csv(
    cleaned_file,
    index=False
)

print("\nCleaned dataset saved.")


# ============================================================
# 10. DESCRIPTIVE STATISTICS
# ============================================================

numeric_descriptive_stats = (
    df
    .select_dtypes(
        include=np.number
    )
    .describe()
    .T
)

numeric_descriptive_stats.to_csv(
    os.path.join(
        output_folder,
        "descriptive_statistics.csv"
    )
)

print("\nDescriptive Statistics:")
print(numeric_descriptive_stats)


# ============================================================
# 11. SUPPLY RISK SUMMARY
# ============================================================

if "Supply_Risk_Flag" in df.columns:

    risk_frequency = (
        df["Supply_Risk_Flag"]
        .value_counts()
        .sort_index()
    )

    risk_percentage = (
        df["Supply_Risk_Flag"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    risk_summary = pd.DataFrame({
        "Frequency": risk_frequency,
        "Percentage": risk_percentage
    })

    risk_summary.to_csv(
        os.path.join(
            output_folder,
            "supply_risk_summary.csv"
        )
    )

    print("\nSupply Risk Summary:")
    print(risk_summary)


# ============================================================
# 12. SUPPLY RISK DISTRIBUTION CHART
# ============================================================

if "Supply_Risk_Flag" in df.columns:

    plt.figure(
        figsize=(8, 5)
    )

    df["Supply_Risk_Flag"] \
        .value_counts() \
        .sort_index() \
        .plot(
            kind="bar"
        )

    plt.title(
        "Supply Risk Distribution"
    )

    plt.xlabel(
        "Supply Risk Flag"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "supply_risk_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 13. DELAY DAYS DISTRIBUTION
# ============================================================

if "Delay_Days" in df.columns:

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        df["Delay_Days"].dropna(),
        bins=20
    )

    plt.title(
        "Distribution of Delay Days"
    )

    plt.xlabel(
        "Delay Days"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "delay_days_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 14. SUPPLIER RELIABILITY DISTRIBUTION
# ============================================================

if "Supplier_Reliability_Score" in df.columns:

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        df[
            "Supplier_Reliability_Score"
        ].dropna(),
        bins=20
    )

    plt.title(
        "Distribution of Supplier Reliability Score"
    )

    plt.xlabel(
        "Supplier Reliability Score"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "supplier_reliability_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 15. DISRUPTION TYPE DISTRIBUTION
# ============================================================

if "Disruption_Type" in df.columns:

    plt.figure(
        figsize=(9, 5)
    )

    df["Disruption_Type"] \
        .value_counts() \
        .plot(
            kind="bar"
        )

    plt.title(
        "Distribution of Disruption Types"
    )

    plt.xlabel(
        "Disruption Type"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "disruption_types.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 16. SHIPPING MODE DISTRIBUTION
# ============================================================

if "Shipping_Mode" in df.columns:

    plt.figure(
        figsize=(8, 5)
    )

    df["Shipping_Mode"] \
        .value_counts() \
        .plot(
            kind="bar"
        )

    plt.title(
        "Shipping Mode Distribution"
    )

    plt.xlabel(
        "Shipping Mode"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "shipping_mode_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 17. CORRELATION ANALYSIS
# ============================================================

numeric_df = df.select_dtypes(
    include=np.number
)

correlation_matrix = (
    numeric_df.corr()
)

correlation_matrix.to_csv(
    os.path.join(
        output_folder,
        "correlation_matrix.csv"
    )
)

print("\nCorrelation Matrix:")
print(correlation_matrix)


# ============================================================
# 18. CORRELATION WITH SUPPLY RISK
# ============================================================

if "Supply_Risk_Flag" in correlation_matrix.columns:

    risk_correlations = (
        correlation_matrix[
            "Supply_Risk_Flag"
        ]
        .sort_values(
            ascending=False
        )
    )

    risk_correlations.to_csv(
        os.path.join(
            output_folder,
            "correlation_with_supply_risk.csv"
        ),
        header=[
            "Correlation_with_Supply_Risk"
        ]
    )

    print(
        "\nCorrelation with Supply Risk:"
    )

    print(
        risk_correlations
    )


# ============================================================
# 19. CORRELATION HEATMAP
# ============================================================

plt.figure(
    figsize=(12, 10)
)

heatmap_data = correlation_matrix.values

plt.imshow(
    heatmap_data,
    aspect="auto"
)

plt.colorbar(
    label="Correlation Coefficient"
)

plt.xticks(
    range(
        len(
            correlation_matrix.columns
        )
    ),
    correlation_matrix.columns,
    rotation=90,
    fontsize=7
)

plt.yticks(
    range(
        len(
            correlation_matrix.index
        )
    ),
    correlation_matrix.index,
    fontsize=7
)

plt.title(
    "Correlation Heatmap of Numerical Variables"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "correlation_heatmap.png"
    ),
    dpi=300
)

plt.close()

print(
    "\nCorrelation heatmap created."
)


# ============================================================
# 20. MULTIPLE LINEAR REGRESSION
# Predict Delay Days
# ============================================================

regression_features = [
    "Supplier_Reliability_Score",
    "Historical_Disruption_Count",
    "Available_Historical_Records",
    "Parameter_Change_Magnitude",
    "Communication_Cost_MB",
    "Energy_Consumption_Joules",
    "Quantity_Ordered",
    "Order_Value_USD"
]

regression_features = [
    feature
    for feature in regression_features
    if feature in df.columns
]


if (
    "Delay_Days" in df.columns
    and len(
        regression_features
    ) > 0
):

    X_reg = df[
        regression_features
    ].copy()

    y_reg = df[
        "Delay_Days"
    ].copy()

    X_reg_constant = sm.add_constant(
        X_reg
    )

    regression_model = sm.OLS(
        y_reg,
        X_reg_constant
    ).fit()

    regression_results = pd.DataFrame({
        "Coefficient":
            regression_model.params,

        "Standard_Error":
            regression_model.bse,

        "T_Value":
            regression_model.tvalues,

        "P_Value":
            regression_model.pvalues
    })

    regression_results.to_csv(
        os.path.join(
            output_folder,
            "multiple_regression_results.csv"
        )
    )

    with open(
        os.path.join(
            output_folder,
            "regression_model_summary.txt"
        ),
        "w"
    ) as file:

        file.write(
            regression_model
            .summary()
            .as_text()
        )

    regression_model_statistics = pd.DataFrame({
        "Statistic": [
            "R_Squared",
            "Adjusted_R_Squared",
            "F_Statistic",
            "F_Test_P_Value",
            "Observations"
        ],

        "Value": [
            regression_model.rsquared,
            regression_model.rsquared_adj,
            regression_model.fvalue,
            regression_model.f_pvalue,
            regression_model.nobs
        ]
    })

    regression_model_statistics.to_csv(
        os.path.join(
            output_folder,
            "regression_model_statistics.csv"
        ),
        index=False
    )

    print(
        "\nMULTIPLE REGRESSION RESULTS"
    )

    print(
        regression_results
    )

    print(
        "\nR Squared:",
        regression_model.rsquared
    )

    print(
        "Adjusted R Squared:",
        regression_model.rsquared_adj
    )

    print(
        "F-statistic:",
        regression_model.fvalue
    )

    print(
        "F-test p-value:",
        regression_model.f_pvalue
    )


# ============================================================
# 21. DEFINE CLASSIFICATION FEATURES
# ============================================================

classification_features = [
    "Supplier_Reliability_Score",
    "Historical_Disruption_Count",
    "Available_Historical_Records",
    "Parameter_Change_Magnitude",
    "Communication_Cost_MB",
    "Energy_Consumption_Joules",
    "Quantity_Ordered",
    "Order_Value_USD"
]

classification_features = [
    feature
    for feature in classification_features
    if feature in df.columns
]


# ============================================================
# 22. PREPARE MACHINE LEARNING DATA
# ============================================================

if (
    "Supply_Risk_Flag" in df.columns
    and len(
        classification_features
    ) > 0
):

    X = df[
        classification_features
    ].copy()

    y = df[
        "Supply_Risk_Flag"
    ].copy()


    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        "\nTraining records:",
        len(
            X_train
        )
    )

    print(
        "Testing records:",
        len(
            X_test
        )
    )


    # --------------------------------------------------------
    # STANDARDIZE DATA FOR LOGISTIC REGRESSION
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # ========================================================
    # 23. CREATE MODELS
    # ========================================================

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    decision_tree_model = DecisionTreeClassifier(
        random_state=42
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )


    # ========================================================
    # 24. TRAIN MODELS
    # ========================================================

    logistic_model.fit(
        X_train_scaled,
        y_train
    )

    decision_tree_model.fit(
        X_train,
        y_train
    )

    random_forest_model.fit(
        X_train,
        y_train
    )


    # ========================================================
    # 25. PREDICTIONS
    # ========================================================

    logistic_predictions = (
        logistic_model.predict(
            X_test_scaled
        )
    )

    decision_tree_predictions = (
        decision_tree_model.predict(
            X_test
        )
    )

    random_forest_predictions = (
        random_forest_model.predict(
            X_test
        )
    )


    # Probabilities for ROC-AUC

    logistic_probabilities = (
        logistic_model
        .predict_proba(
            X_test_scaled
        )[:, 1]
    )

    decision_tree_probabilities = (
        decision_tree_model
        .predict_proba(
            X_test
        )[:, 1]
    )

    random_forest_probabilities = (
        random_forest_model
        .predict_proba(
            X_test
        )[:, 1]
    )


    # ========================================================
    # 26. EVALUATION FUNCTION
    # ========================================================

    def evaluate_model(
        model_name,
        actual,
        predictions,
        probabilities
    ):

        accuracy = accuracy_score(
            actual,
            predictions
        )

        precision = precision_score(
            actual,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            actual,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            actual,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            actual,
            probabilities
        )

        return {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "ROC_AUC": roc_auc
        }


    # ========================================================
    # 27. MODEL COMPARISON
    # ========================================================

    results = []

    results.append(
        evaluate_model(
            "Logistic Regression",
            y_test,
            logistic_predictions,
            logistic_probabilities
        )
    )

    results.append(
        evaluate_model(
            "Decision Tree",
            y_test,
            decision_tree_predictions,
            decision_tree_probabilities
        )
    )

    results.append(
        evaluate_model(
            "Random Forest",
            y_test,
            random_forest_predictions,
            random_forest_probabilities
        )
    )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        os.path.join(
            output_folder,
            "model_comparison.csv"
        ),
        index=False
    )

    print(
        "\nMODEL COMPARISON"
    )

    print(
        results_df
    )


    # ========================================================
    # 28. SAVE CLASSIFICATION REPORTS
    # ========================================================

    classification_models = {
        "logistic_regression":
            logistic_predictions,

        "decision_tree":
            decision_tree_predictions,

        "random_forest":
            random_forest_predictions
    }

    for name, predictions in classification_models.items():

        report = classification_report(
            y_test,
            predictions,
            zero_division=0
        )

        with open(
            os.path.join(
                output_folder,
                f"{name}_classification_report.txt"
            ),
            "w"
        ) as file:

            file.write(
                report
            )


    # ========================================================
    # 29. CONFUSION MATRIX FUNCTION
    # ========================================================

    def save_confusion_matrix(
        actual,
        predictions,
        title,
        filename
    ):

        cm = confusion_matrix(
            actual,
            predictions
        )

        plt.figure(
            figsize=(6, 5)
        )

        plt.imshow(
            cm
        )

        plt.title(
            title
        )

        plt.xlabel(
            "Predicted"
        )

        plt.ylabel(
            "Actual"
        )

        plt.xticks(
            [0, 1],
            [
                "No Risk",
                "Risk"
            ]
        )

        plt.yticks(
            [0, 1],
            [
                "No Risk",
                "Risk"
            ]
        )

        for i in range(
            cm.shape[0]
        ):

            for j in range(
                cm.shape[1]
            ):

                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_folder,
                filename
            ),
            dpi=300
        )

        plt.close()


    # Logistic Regression confusion matrix

    save_confusion_matrix(
        y_test,
        logistic_predictions,
        "Logistic Regression Confusion Matrix",
        "logistic_regression_confusion_matrix.png"
    )


    # Decision Tree confusion matrix

    save_confusion_matrix(
        y_test,
        decision_tree_predictions,
        "Decision Tree Confusion Matrix",
        "decision_tree_confusion_matrix.png"
    )


    # Random Forest confusion matrix

    save_confusion_matrix(
        y_test,
        random_forest_predictions,
        "Random Forest Confusion Matrix",
        "random_forest_confusion_matrix.png"
    )


    # ========================================================
    # 30. ROC CURVE COMPARISON
    # ========================================================

    fpr_lr, tpr_lr, _ = roc_curve(
        y_test,
        logistic_probabilities
    )

    fpr_dt, tpr_dt, _ = roc_curve(
        y_test,
        decision_tree_probabilities
    )

    fpr_rf, tpr_rf, _ = roc_curve(
        y_test,
        random_forest_probabilities
    )


    auc_lr = auc(
        fpr_lr,
        tpr_lr
    )

    auc_dt = auc(
        fpr_dt,
        tpr_dt
    )

    auc_rf = auc(
        fpr_rf,
        tpr_rf
    )


    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr_lr,
        tpr_lr,
        label=f"Logistic Regression (AUC = {auc_lr:.3f})"
    )

    plt.plot(
        fpr_dt,
        tpr_dt,
        label=f"Decision Tree (AUC = {auc_dt:.3f})"
    )

    plt.plot(
        fpr_rf,
        tpr_rf,
        label=f"Random Forest (AUC = {auc_rf:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier (AUC = 0.500)"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve Comparison of Predictive Models"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "roc_curve_comparison.png"
        ),
        dpi=300
    )

    plt.close()


    print(
        "\nROC CURVE RESULTS"
    )

    print(
        "Logistic Regression AUC:",
        round(
            auc_lr,
            3
        )
    )

    print(
        "Decision Tree AUC:",
        round(
            auc_dt,
            3
        )
    )

    print(
        "Random Forest AUC:",
        round(
            auc_rf,
            3
        )
    )


    # ========================================================
    # 31. FIVE-FOLD CROSS-VALIDATION
    # ========================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_models = {

        "Logistic Regression":
            Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42
                    )
                )
            ]),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
    }


    cv_results = []


    for model_name, model in cv_models.items():

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )

        cv_results.append({

            "Model":
                model_name,

            "Mean_CV_Accuracy":
                scores.mean(),

            "SD_CV_Accuracy":
                scores.std(),

            "Fold_1":
                scores[0],

            "Fold_2":
                scores[1],

            "Fold_3":
                scores[2],

            "Fold_4":
                scores[3],

            "Fold_5":
                scores[4]
        })


    cv_results_df = pd.DataFrame(
        cv_results
    )


    cv_results_df.to_csv(
        os.path.join(
            output_folder,
            "cross_validation_results.csv"
        ),
        index=False
    )


    print(
        "\n5-FOLD CROSS-VALIDATION RESULTS"
    )

    print(
        cv_results_df
    )


    # ========================================================
    # 32. RANDOM FOREST FEATURE IMPORTANCE
    # ========================================================

    feature_importance = pd.DataFrame({

        "Feature":
            classification_features,

        "Importance":
            random_forest_model
            .feature_importances_
    })


    feature_importance = (
        feature_importance
        .sort_values(
            by="Importance",
            ascending=False
        )
    )


    feature_importance.to_csv(
        os.path.join(
            output_folder,
            "random_forest_feature_importance.csv"
        ),
        index=False
    )


    print(
        "\nRANDOM FOREST FEATURE IMPORTANCE"
    )

    print(
        feature_importance
    )


    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        feature_importance[
            "Feature"
        ],
        feature_importance[
            "Importance"
        ]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "random_forest_feature_importance.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 33. CREATE ANALYSIS SUMMARY FILE
# ============================================================

summary_text = """
SUPPLY CHAIN PREDICTIVE ANALYTICS ANALYSIS COMPLETED

FILES GENERATED:

DATA CLEANING
-------------
1. supply_chain_cleaned.csv
2. missing_values.csv

DESCRIPTIVE ANALYSIS
--------------------
3. descriptive_statistics.csv
4. supply_risk_summary.csv
5. supply_risk_distribution.png
6. delay_days_distribution.png
7. supplier_reliability_distribution.png
8. disruption_types.png
9. shipping_mode_distribution.png

CORRELATION ANALYSIS
--------------------
10. correlation_matrix.csv
11. correlation_with_supply_risk.csv
12. correlation_heatmap.png

REGRESSION ANALYSIS
-------------------
13. multiple_regression_results.csv
14. regression_model_statistics.csv
15. regression_model_summary.txt

MACHINE LEARNING
----------------
16. model_comparison.csv
17. logistic_regression_classification_report.txt
18. decision_tree_classification_report.txt
19. random_forest_classification_report.txt

CONFUSION MATRICES
------------------
20. logistic_regression_confusion_matrix.png
21. decision_tree_confusion_matrix.png
22. random_forest_confusion_matrix.png

MODEL VALIDATION
----------------
23. roc_curve_comparison.png
24. cross_validation_results.csv

FEATURE IMPORTANCE
------------------
25. random_forest_feature_importance.csv
26. random_forest_feature_importance.png

All results are stored in:
supply_chain_analysis_results
"""


with open(
    os.path.join(
        output_folder,
        "analysis_summary.txt"
    ),
    "w"
) as file:

    file.write(
        summary_text
    )


# ============================================================
# 34. FINISHED
# ============================================================

print(
    "\n=========================================="
)

print(
    "ANALYSIS COMPLETED SUCCESSFULLY"
)

print(
    "=========================================="
)

print(
    "\nAll results are saved inside:"
)

print(
    output_folder
)