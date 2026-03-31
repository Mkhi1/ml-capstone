# Titanic Survival Analysis
## ML Foundations Bootcamp — Capstone Report

**Student:** 
**Dataset:** Titanic Passenger Dataset (891 records)
**Date:** March 2026
**Notebooks:** `01_cleaning.ipynb` · `02_features.ipynb` · `03_eda.ipynb` · `04_math.ipynb`

---

## 1. Introduction

For this capstone project I worked with the **Titanic dataset**, which contains records of 891 passengers aboard the RMS Titanic when it sank on 15 April 1912 after striking an iceberg in the North Atlantic. The ship carried 2,224 people but only 710 lifeboats seats were available — meaning survival was, in part, a matter of chance, but also heavily shaped by social and economic factors of the era.

Each row in the dataset represents one passenger and includes columns such as:

| Column | What it means |
|---|---|
| `Survived` | 1 = survived, 0 = did not survive (target variable) |
| `Pclass` | Ticket class: 1st, 2nd, or 3rd |
| `Sex` | Male or female |
| `Age` | Passenger age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Ticket price paid |
| `Embarked` | Port of embarkation: S=Southampton, C=Cherbourg, Q=Queenstown |

**Research questions I tried to answer:**

1. Which factors had the strongest influence on survival?
2. Did wealth (ticket class, fare) predict survival?
3. Did gender play a bigger role than age or class?
4. Did family size affect survival chances?

---

## 2. Data Cleaning

### 2.1 Initial Data State

When I first loaded the raw CSV with `pd.read_csv()`, the dataset had **891 rows and 12 columns**. Running `.isnull().sum()` revealed three columns with missing data:

| Column | Missing Values | % Missing | Action Taken |
|---|---|---|---|
| `Cabin` | 687 | 77% | **Dropped** — too sparse to fill reliably |
| `Age` | 177 | 20% | **Filled with median (28)** — median is more robust than mean for skewed data |
| `Embarked` | 2 | 0.2% | **Filled with mode ("S")** — only 2 rows missing, mode is safe |

### 2.2 Data Type Fixes

Columns `Survived`, `Pclass`, `Sex`, and `Embarked` were stored as integers or plain strings, but they represent categories. I converted them to the correct types using `.astype('category')`. This prevents pandas from treating class 3 as mathematically "three times" class 1.

### 2.3 Outlier Handling

The `Fare` column had a maximum of **£512** while the median was only **£14**. This extreme right skew was caused by a small number of very expensive cabin bookings. I used the **99th percentile capping** (Winsorization) method: any fare above the 99th percentile was capped at that value. This kept all 891 rows while reducing the distortion from extreme values.

```
Before capping: Fare max = 512.33
After capping:  Fare max = 249.00
Rows affected:  9 passengers (1% of data)
```

### 2.4 clean_data() Function & Validation

All cleaning steps were wrapped into a single reusable `clean_data()` function. After running it, I added three assertions to confirm the data was clean:

- ✅ No nulls in `Age`, `Embarked`, `Sex`, or `Fare`
- ✅ All `Survived` values are exactly 0 or 1
- ✅ `Cabin` column no longer exists

**Final cleaned dataset:** 891 rows × 11 columns, saved to `data/cleaned/titanic_cleaned.csv`.

---

## 3. Feature Engineering

### 3.1 Encoding Categorical Columns

| Technique | Applied to | Why |
|---|---|---|
| One-hot encoding | `Sex`, `Embarked` | No natural order — each category becomes its own 0/1 column |
| Ordinal (kept as-is) | `Pclass` (1, 2, 3) | Order is meaningful: 1st class genuinely better than 3rd |

### 3.2 Scaling

`Age` and `Fare` were standardised with `StandardScaler` (z = (x − mean) / std). Without scaling, Fare values (0–512) would dominate Age values (0–80) purely because of their larger numeric range.

### 3.3 New Domain Features

| Feature | Formula | Why it's useful |
|---|---|---|
| `FamilySize` | SibSp + Parch + 1 | Total group size — travelling alone vs. with family changes survival odds |
| `FarePerPerson` | Fare / FamilySize | Normalises fare: a family of 5 paying £50 together is different from one person paying £50 |
| `IsAlone` | 1 if FamilySize == 1 | Binary flag — easier for models than a continuous family size count |
| `Class_Fare` | Pclass × Fare | Interaction feature — captures combined effect of class and fare price |

### 3.4 Transformations & Binning

- **Log transform on Fare:** Applied `np.log1p(Fare)` to compress the right-skewed distribution into a near-normal shape. The histogram before/after clearly shows the improvement.
- **Age binning:** Age was grouped into 5 meaningful bands:

| Age Group | Range | Count |
|---|---|---|
| Child | 0–12 | 69 |
| Teen | 13–18 | 52 |
| YoungAdult | 19–35 | 335 |
| Adult | 36–60 | 200 |
| Senior | 61+ | 22 |

After feature engineering the dataset grew from 11 to **18 columns** and was saved to `data/cleaned/titanic_features.csv`.

---

## 4. Key Findings

### Finding 1 — Gender Was the Single Biggest Factor

The survival rate split by sex was the most dramatic gap in the entire dataset:

| Sex | Survived | Did Not Survive | Survival Rate |
|---|---|---|---|
| **Female** | 233 | 81 | **74%** |
| **Male** | 109 | 468 | **19%** |

Women were **nearly 4× more likely to survive** than men. This directly reflects the "women and children first" evacuation policy enforced by the crew. Of the 342 survivors, 233 (68%) were women, even though women made up only 35% of all passengers.

> **Chart reference:** See *Survival Rate by Sex* bar chart in the dashboard figure.

---

### Finding 2 — Passenger Class Was Almost as Powerful as Gender

Survival rates decreased sharply with each step down in class:

| Class | Passengers | Survivors | Survival Rate |
|---|---|---|---|
| 1st | 216 | 136 | **63%** |
| 2nd | 184 | 87 | **47%** |
| 3rd | 491 | 119 | **24%** |

A 1st class passenger was **2.6× more likely to survive** than a 3rd class passenger. Likely reasons: 1st class cabins were on higher decks closer to the lifeboats, and historical accounts suggest that crew prioritised wealthier passengers during the evacuation.

> **Chart reference:** See *Survival Rate by Class* bar chart in the dashboard figure.

---

### Finding 3 — Fare Distribution Is Heavily Right-Skewed

The histogram of raw `Fare` shows that the vast majority of passengers paid under £30, with a long right tail reaching £512. This skew is a classic real-world data problem — a few very wealthy first-class passengers paid extraordinarily high prices that distort the average.

After applying `np.log1p()`, the distribution became near-normal and much easier to analyse. The correlation heatmap also confirmed that `Fare` had one of the highest positive correlations with `Survived` (r ≈ +0.26), meaning higher fares correlated with better survival odds.

> **Chart reference:** See *Age Distribution* and *Fare by Class* charts in the dashboard figure.

---

### Best Chart — Dashboard: Four Key Views in One

The dashboard below (saved as `dashboard.png`) shows the four most important patterns in a single figure:

- **Top-left:** Survival rate by passenger class — clear step-down from 1st to 3rd class
- **Top-right:** Survival rate by sex — the largest gap in the dataset
- **Bottom-left:** Age distribution — most passengers were young adults (19–35)
- **Bottom-right:** Fare boxplot by class — 1st class passengers paid dramatically more, confirming the link between wealth and survival

*(See `dashboard.png` in the project folder)*

---

## 5. Math Basics Summary

In `04_math.ipynb` I applied the following mathematical operations directly to the data:

**Manual mean & standard deviation (NumPy):**
```
Target column: Survived
Mean (NumPy):       0.3838 — about 38% of passengers survived
Std Dev (NumPy):    0.4866
```

**Manual standardisation (broadcasting):**
```
z = (Fare - mean) / std
Max difference from StandardScaler: 0.0000 (perfect match)
```
This confirmed that my manual broadcasting formula produces identical results to sklearn's StandardScaler.

**Cosine similarity:**
Computed between the feature vectors of the passenger with the highest Fare and the passenger with the lowest Fare. The cosine similarity was low, confirming these two passengers have very different feature profiles.

**Probability estimate:**
```
P(survived | Pclass == 1): 0.630 (63%)
P(survived | Pclass == 3): 0.242 (24%)
```
This matches the groupby findings from the EDA and confirms the class-based survival disparity.

---

## 6. What I Would Do Next

1. **Build a classification model** — the data is now clean and feature-engineered, making it ready for a decision tree, logistic regression, or random forest model to predict survival. Given the strong signals from gender and class, a simple logistic regression could likely reach 78–80% accuracy.

2. **Extract titles from the Name column** — names like "Mr", "Mrs", "Miss", "Dr", and "Master" encode both gender and social status. Extracting these titles as a feature could improve model performance.

3. **Analyse family survival clusters** — families (grouped by last name and ticket) likely had correlated outcomes. If a mother survived, her children probably did too. Analysing group-level survival patterns could reveal social dynamics not visible at the individual level.

4. **Submit to Kaggle** — the Titanic dataset has a public leaderboard on Kaggle. Submitting predictions would provide an objective external benchmark for the quality of this analysis.

---

## 7. Conclusion

The Titanic dataset tells a clear and sobering story: survival in the disaster was not random. It was heavily shaped by **gender** (women first), **wealth** (1st class first), and **age** (children first). The data cleaning process removed significant noise from the raw data — missing values, wrong types, and outlier fares — and the feature engineering step created new columns that made these patterns easier to see and model. The exploratory analysis, statistical tests, and visualisations all point to the same conclusion:

> **A woman in 1st class had roughly a 95% chance of survival. A man in 3rd class had less than 15%.**

This project demonstrates how a complete data analysis pipeline — from raw CSV to clean insights — can turn historical records into actionable understanding.

---

*All code is in `01_cleaning.ipynb`, `02_features.ipynb`, `03_eda.ipynb`, and `04_math.ipynb`.
Cleaned data is in `data/cleaned/`. Charts are generated when notebooks are run.*
