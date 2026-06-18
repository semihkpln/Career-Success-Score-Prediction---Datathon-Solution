# Career-Success-Score-Prediction---Datathon-Solution
This repository contains the machine learning solution developed for a Datathon competition aimed at predicting the Career Success Score of students and recent graduates.   The final model is a robust Stacking &amp; Blending Ensemble architecture built upon advanced feature engineering and a strict cross-validation strategy to prevent overfitting.
## 📊 Project Overview
The dataset provided in the competition includes candidates' technical skills, interview performances, university tiers, and mentor feedback. The primary objective is to use these parameters to predict the candidate's career success score with the lowest possible Mean Squared Error (MSE).

## 🛠️ Solution Architecture & Approach

The project goes beyond basic algorithms, employing the following advanced data science techniques:

* **Comprehensive Feature Engineering:** Blended existing variables to derive over 35 new logical features (e.g., the interaction between project quality and technical interview scores, and the balance between tech vs. soft skills).
* **Data Leakage Prevention:** Categorical variables (such as "target_role" and "department") were encoded using **Out-of-Fold (OOF) Target Encoding**. This strict cross-validation approach ensures the model generalizes well and prevents overfitting.
* **Natural Language Processing (NLP):** Mentor feedback texts were processed using rule-based sentiment analysis to prevent potential data leakage, or optionally mapped into a vector space using the **BERTürk** deep learning language model.
* **Hyperparameter Optimization:** Base models including XGBoost, LightGBM, CatBoost, and ExtraTrees were heavily optimized using the **Optuna** framework.
* **Ensemble Modeling:** Predictions from the base models were dynamically weighted using a Meta-Learner (**Ridge Regression Stacking**) and **Inverse-MSE Blending** to achieve the final, highly accurate predictions.

## 💻 Technologies Used

* **Language:** Python
* **Machine Learning:** Scikit-learn, XGBoost, LightGBM, CatBoost, ExtraTrees
* **Optimization:** Optuna
* **NLP & Deep Learning:** PyTorch, HuggingFace (BERTürk)
* **Data Manipulation:** Pandas, NumPy

## 📂 Folder Structure

* `train.csv` & `test_x.csv` : Competition datasets.
* `model_2.3.py` : The main solution script containing data preprocessing, OOF target encoding, hyperparameter optimization, and the final Ensemble architecture.
* `submission_2.3.csv` : The final predictions generated for the competition.

---

## 📬 Contact

**Semih Kaplan**

* **Email:** msemihkpln@gmail.com
* **GitHub:** [github.com/semihkpln](https://github.com/semihkpln)
