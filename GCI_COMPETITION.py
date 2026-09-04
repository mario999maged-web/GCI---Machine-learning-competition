# Import modules
import numpy as np  # Library for numerical computing and array operations
import pandas as pd  # Library for handling tabular data
import matplotlib.pyplot as plt  # Library for basic data visualization
import seaborn as sns  # Library for statistical data visualization
from sklearn.preprocessing import LabelEncoder  # Encoder for converting categorical variables to numeric labels
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier
from sklearn.model_selection import StratifiedKFold  # Class for stratified K-fold cross-validation
from sklearn.metrics import roc_auc_score  # Metric function for computing ROC AUC

'''
train = pd.read_csv('train.csv')  # Load training data from CSV file  
test = pd.read_csv('test.csv')  # Load test data from CSV file

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("*" * 50)
print(train.columns)
#print('train info',train.info())
#print('\ntest info',test.info())

print("Missing values in train data:\n", train.isnull().sum())  # Check for missing values in the training data


# 1. حساب نسبة النجاح لكل مدرسة في الـ train
school_success = train.groupby('School')['Drafted'].mean()

for df in [train, test]:
    # بنباصي النسبة دي للـ train والـ test
    df['School_Success_Rate'] = df['School'].map(school_success).fillna(0)


    # بنعمل نسبة بين الـ Shuttle بتاع اللاعب ومتوسط Shuttle لكل Position (عشان نعرف إذا كان Shuttle بتاعه كويس بالنسبة لزملائه في نفس المركز ولا لأ)
df['Shuttle_vs_Pos'] = df['Shuttle'] / df.groupby('Position')['Shuttle'].transform('mean')
import matplotlib.pyplot as plt

# تحديد الأعمدة اللي عايز ترسمها
cat_cols = ['Player_Type', 'Position_Type', 'Position']

# إنشاء الشكل (3 رسمات جنب بعض)
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
drafted_only = train[train['Drafted'] == 1].copy()
for i, col in enumerate(cat_cols):
    # حساب النسب
    counts = drafted_only[col].value_counts(normalize=True)
    
    # تجميع الفئات الصغيرة في "Other" (نظافة للرسمة)
    threshold = 0.03
    small = counts[counts < threshold].sum()
    counts = counts[counts >= threshold]
    if small > 0:
        counts['Other'] = small
        
    # الرسم
    axes[i].pie(
        counts, 
        labels=counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        pctdistance=0.85, 
        colors=plt.cm.Paired.colors
    )
    
    # تحويلها لـ Donut (الدائرة البيضاء في النص)
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    axes[i].add_artist(centre_circle)
    
    # تظبيط العنوان
    axes[i].set_title(f'{col} Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
print(plt.show())


model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=2025)

# 2. تجهيز نسخة من البيانات عشان نلعب فيها براحتنا
X_plot = train.copy()

# 3. صياعة المحترفين: تحويل الأعمدة النصية لأرقام (عشان ما ندمرهاش)
categorical_cols = ['Player_Type', 'Position_Type', 'Position', 'School']

for col in categorical_cols:
    if col in X_plot.columns:
        # astype('category').cat.codes بيحول كل نص لرقم فريد تلقائياً (0, 1, 2...)
        X_plot[col] = X_plot[col].astype('category').cat.codes

# 4. دلوقتي نحذف بس الأعمدة اللي بجد ملهاش عازة (زي اسم اللاعب والـ Id والـ Target)
X_plot = X_plot.drop(columns=['Drafted', 'Id', 'Player_Name'], errors='ignore')

# 5. املأ الفراغات (بما فيهم الـ NaNs اللي ظهرت لو فيه سن مفقود)
X_plot = X_plot.fillna(-999)
y = train['Drafted']

# 6. تدريب الموديل (دلوقتي هيقرا كله!)
model.fit(X_plot, y)

# 7. رسم الـ Importance الشاملة
importances = model.feature_importances_
feature_names = X_plot.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importance - الترتيب العادل لكل الميزات")
plt.bar(range(X_plot.shape[1]), importances[indices], align="center")
plt.xticks(range(X_plot.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
'''

import os
from pathlib import Path

current_dir = Path(os.getcwd())
client_path = current_dir / "telecom" / "Client.csv"
record_path = current_dir / "telecom" / "Record.csv"

for p in (client_path, record_path):
    print(f"{'OK ' if p.exists() else 'MISSING '} {p}")


client = pd.read_csv(client_path)
record = pd.read_csv(record_path)

print(f"Client: {client.shape}")
print(f"Record: {record.shape}")





































