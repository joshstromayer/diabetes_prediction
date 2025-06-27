import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# df = pd.read_csv("diabetes_prediction_dataset.csv")

# Encoding dataset to usable data
"""
df["smoking_history"] = df["smoking_history"].replace('No Info', np.nan)
is_null = df.isnull().sum() 
# print(is_null) # 35816 null values in smoking_history, 0 in rest. 

df = pd.get_dummies(df, columns=["gender"])

encoded_data = pd.get_dummies(df, columns=['smoking_history'])
encoded_data.to_csv("encoded_data.csv", index=False)"""

df = pd.read_csv("encoded_data.csv")

# corr = df.corr(numeric_only=True)
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr, annot=True, fmt = '.2f')
# plt.title("Correlation Heatmap: All Features")
# plt.tight_layout()
# plt.savefig("correlation_heatmap_all_features.png")
# plt.show()

# smoking_history and gender columns need to go, very weak correlation 

df.drop(['gender_Female', 'gender_Male', 'gender_Other', 'smoking_history_current', 'smoking_history_ever', 'smoking_history_former', 'smoking_history_never', 'smoking_history_not current'], axis=1, inplace=True)

# corr = df.corr(numeric_only=True)
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr, annot=True, fmt = '.2f')
# plt.title("Correlation Heatmap: Selective Features")
# plt.tight_layout()
# plt.savefig("correlation_heatmaps_selective_features.png")
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.set_palette(['#a78cd9', '#cda4de'])
# sns.set_style("ticks") 

# plt.title("Blood Glucose Levels by Diabetes Status")
# bp = sns.boxplot(data=df, x='diabetes', y='blood_glucose_level', width=0.7)
# bp.set_xticklabels(['Non-Diabetic', 'Diabetic'])
# plt.xlabel("Diabetes")
# plt.ylabel("Blood Glucose Levels")
# plt.savefig("images/boxplot_bgl.png")
# plt.show()

# plt.title("HbA1c Levels by Diabetes Status")
# bp = sns.boxplot(data=df, x='diabetes', y='HbA1c_level', width=0.7)
# bp.set_xticklabels(['Non-Diabetic', 'Diabetic'])
# plt.xlabel("Diabetes")
# plt.ylabel("HbA1c")
# plt.savefig("images/boxplot_hbA1c.png")
# plt.show()

# plt.title("BMI by Diabetes Status")
# bp = sns.boxplot(data=df, x='diabetes', y='bmi', width=0.7)
# bp.set_xticklabels(['Non-Diabetic', 'Diabetic'])
# plt.xlabel("Diabetes")
# plt.ylabel("BMI")
# plt.savefig("images/boxplot_bmi.png")
# plt.show()

# plt.title("Age by Diabetes Status")
# bp = sns.boxplot(data=df, x='diabetes', y='age', width=0.7)
# bp.set_xticklabels(['Non-Diabetic', 'Diabetic'])
# plt.xlabel("Diabetes")
# plt.ylabel("Age")
# plt.savefig("images/boxplot_age.png")
# plt.show()

df = pd.read_csv("encoded_data.csv")

# fig, axs = plt.subplots(1, 2)
# axs[0].set_title("Age Distribution")
# axs[0].set_xlabel("Age")
# axs[0].set_ylabel("Frequency Density")
# sns.histplot(data=df[df['diabetes'] == 1]['age'], label='Diabetic', bins=20, color='#a78cd9', ax=axs[0], stat='density')
# sns.histplot(data=df[df['diabetes'] == 0]['age'], label='Non-Diabetic', bins=20, color='#72bad5', ax=axs[0], stat='density')

# axs[1].set_title("BMI Distribution")
# axs[1].set_xlabel("BMI")
# axs[1].set_ylabel('Frequency Density')
# sns.histplot(data=df[df['diabetes'] == 0]['bmi'], label='Non-Diabetic', bins=20, color='#72bad5', ax=axs[1], stat='density')
# sns.histplot(data=df[df['diabetes'] == 1]['bmi'], label='Diabetic', bins=20, color='#a78cd9', ax=axs[1], stat='density')
# plt.tight_layout(pad=2) 
# plt.legend()
# plt.savefig("images/age_and_bmi_dense_histograms.png")
# plt.show()

# fig, axs = plt.subplots(1, 2)
# axs[0].set_title("Blood Glucose Levels Distribution")
# axs[0].set_xlabel("Blood Glucose Levels")
# axs[0].set_ylabel("Frequency Density")
# sns.violinplot(data=df[df['diabetes'] == 1]['blood_glucose_level'], label='Diabetic', color='#a78cd9', ax=axs[0])
# sns.violinplot(data=df[df['diabetes'] == 0]['blood_glucose_level'], label='Non-Diabetic', color='#72bad5', ax=axs[0])

# axs[1].set_title("HbA1c Levels Distribution")
# axs[1].set_xlabel("HbA1c Levels")
# axs[1].set_ylabel('Frequency Density')
# sns.violinplot(data=df[df['diabetes'] == 0]['HbA1c_level'], label='Non-Diabetic', color='#72bad5', ax=axs[1])
# sns.violinplot(data=df[df['diabetes'] == 1]['HbA1c_level'], label='Diabetic', color='#a78cd9', ax=axs[1])
# plt.tight_layout(pad=2) 
# plt.legend()
# plt.savefig("images/bgl_and_hb1ac_dense_violinplots.png")
# plt.show()

