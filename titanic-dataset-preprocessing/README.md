# Machine Learning Data Preprocessing on the Titanic Dataset

## Overview

This project presents a complete data preprocessing pipeline applied to the classic Titanic passenger dataset. In machine learning workflows, raw data is rarely ready for model training. Real-world datasets often suffer from missing entries, extreme outliers, non-standard numerical distributions, categorical variables, and class imbalances.

The goal of this work is to systematically address each of these challenges to produce a clean, balanced, and well-scaled feature set ready for downstream classification algorithms.

---

## Dataset Description

The dataset consists of passenger records from the RMS Titanic. The primary objective is to predict whether a passenger survived (`Survived = 1`) or perished (`Survived = 0`).

Key attributes included in the raw dataset:
* PassengerId: Unique identifier for each traveler
* Survived: Target variable (0 = No, 1 = Yes)
* Pclass: Ticket class (1st, 2nd, 3rd)
* Name: Full passenger name including titles (Mr., Mrs., Miss., Master, etc.)
* Sex: Passenger gender
* Age: Passenger age in years
* SibSp: Number of siblings or spouses aboard
* Parch: Number of parents or children aboard
* Ticket: Ticket number string
* Fare: Passenger fare in British pounds
* Cabin: Cabin number
* Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---

## Preprocessing Steps

### 1. Data Cleaning and Missing Value Imputation

Inspection of the dataset revealed missing values across three primary features:
* **Age**: Missing in approximately 20% of rows. Because age is skewed by children and elderly individuals, replacing null values with the median (28.0) is more robust than using the mean, which is sensitive to extreme values. The values were subsequently rounded and cast to integer format.
* **Cabin**: Over 77% of entries are null. Because imputing such a high proportion of missing data would introduce artificial noise, the feature was dropped from the analysis.
* **Embarked**: Missing in only two records. These were filled with the statistical mode (`'S'`, Southampton), which represents the most frequent port of embarkation.

In addition, feature extraction was performed on the `Name` column to pull out social titles (such as Mr, Mrs, Miss, Master, Dr, Rev) using regular expressions. This retains demographic signals prior to discarding raw string identifiers.

---

### 2. Outlier Detection and Capping (IQR Method)

Numerical features, particularly `Fare` and `Age`, contain extreme values that can disproportionately distort linear and distance-based estimators. Rather than deleting these rows and discarding valuable data, the Interquartile Range (IQR) capping method was applied.

For any given feature:
* Lower Bound = Q1 - 1.5 * IQR
* Upper Bound = Q3 + 1.5 * IQR

Values outside these boundaries were clamped to the respective threshold limits.

#### Fare Distribution (Before vs. After Capping)

| Before Outlier Treatment | After Outlier Treatment |
| :---: | :---: |
| ![Fare Before Capping](images/boxplot_fare_before.png) | ![Fare After Capping](images/boxplot_fare_after.png) |

Before capping, several passenger fares exceeded 200 and 500, creating an extreme right tail. Capping brings these extreme points to the upper statistical boundary without discarding the records.

#### Age Distribution (Before vs. After Capping)

| Before Outlier Treatment | After Outlier Treatment |
| :---: | :---: |
| ![Age Before Capping](images/boxplot_age_before.png) | ![Age After Capping](images/boxplot_age_after.png) |

Extreme ages in the upper tail were bounded while preserving the core demographic spread.

#### Family Features: SibSp and Parch

| Feature | Before Capping | After Capping |
| :--- | :---: | :---: |
| **SibSp** | ![SibSp Before](images/boxplot_sibsp_before.png) | ![SibSp After](images/boxplot_sibsp_after.png) |
| **Parch** | ![Parch Before](images/boxplot_parch_before.png) | ![Parch After](images/boxplot_parch_after.png) |

---

### 3. Feature Transformation and Scaling

#### Categorical Encoding
Machine learning algorithms operate on numerical tensors. Categorical attributes were converted using One-Hot Encoding (`pd.get_dummies`):
* `Sex`: Converted into binary dummy columns (`Sex_female`, `Sex_male`).
* `Embarked`: Converted into binary flags (`Embarked_C`, `Embarked_Q`, `Embarked_S`).

#### Scaling and Normalization
Features have fundamentally different units and magnitudes. Two complementary scaling techniques were evaluated:
* **Min-Max Normalization**: Applied to `Fare` to compress values strictly into the `[0, 1]` interval.
* **Standardization (Z-Score)**: Applied to `Age` to center the distribution around mean 0 with a unit standard deviation (variance = 1).

![Feature Scaling Comparison](images/feature_scaling_comparison.png)

The comparative visualization above demonstrates the effect of both transformations:
* Left panels show raw feature distributions before transformation.
* Right panels show the scaled representations, making the features comparable for gradient-based or distance-based algorithms such as Logistic Regression, SVM, and KNN.

---

### 4. Handling Imbalanced Classes with SMOTE

In the raw Titanic dataset, passenger survival is unevenly distributed: approximately 61% of passengers did not survive, compared to 39% who survived. When models train on skewed targets, they often develop a bias toward predicting the majority class.

To remedy this, SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training set. Rather than simple duplicate sampling, SMOTE synthesizes new plausible minority examples along the feature space vectors connecting k-nearest neighbors.

#### Class Distribution Comparison

| Before SMOTE (Imbalanced) | After SMOTE (Balanced) |
| :---: | :---: |
| ![Distribution Before SMOTE](images/target_distribution_before_smote.png) | ![Distribution After SMOTE](images/target_distribution_after_smote.png) |

As shown in the charts, SMOTE equalizes the class representation, providing an unbiased foundation for model evaluation and metric calculation (such as F1-score, Precision, and Recall).

---

## Repository Structure

```text
.
├── .gitignore
├── README.md
├── Titanic-Dataset.csv
├── pr-traitement-d-un-dataset-en-machine-learning.ipynb
└── images/
    ├── boxplot_age_after.png
    ├── boxplot_age_before.png
    ├── boxplot_fare_after.png
    ├── boxplot_fare_before.png
    ├── boxplot_parch_after.png
    ├── boxplot_parch_before.png
    ├── boxplot_sibsp_after.png
    ├── boxplot_sibsp_before.png
    ├── feature_scaling_comparison.png
    ├── target_distribution_after_smote.png
    └── target_distribution_before_smote.png
```

---

## Getting Started

### Prerequisites

You need Python 3.8 or higher installed on your system. The required libraries can be installed using pip:

```bash
pip install numpy pandas scikit-learn seaborn matplotlib imbalanced-learn jupyter
```

### Running the Notebook

1. Clone or download this repository to your local machine:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. Launch Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook pr-traitement-d-un-dataset-en-machine-learning.ipynb
   ```

3. Execute the cells sequentially from top to bottom. The notebook will automatically read `Titanic-Dataset.csv` from the project directory.

---

## Conclusion

This preprocessing pipeline demonstrates the essential steps required to prepare raw tabular data for predictive modeling:
1. Missing value handling tailored to feature distribution and percentage of nulls.
2. IQR capping to mitigate extreme outlier influence without losing sample volume.
3. Proper categorical encoding and appropriate scaling techniques (Min-Max vs. Standard Scaler).
4. Addressing target imbalance using synthetic oversampling via SMOTE.
