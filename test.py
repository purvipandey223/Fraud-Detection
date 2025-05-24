import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import itertools
from collections import Counter
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
df = pd.read_csv('./Fraud.csv')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
new_df = df.copy()
objList = new_df.select_dtypes(include = 'object').columns
for feat in objList:
    new_df[feat] = le.fit_transform(new_df[feat].astype(str))


scaler = StandardScaler()
new_df["NormalizedAmount"] = scaler.fit_transform(new_df["amount"].values.reshape(-1, 1))
new_df.drop(["amount"], inplace= True, axis= 1)

Y = new_df["isFraud"]
X = new_df.drop(["isFraud"], axis= 1)

(X_train, X_test, Y_train, Y_test) = train_test_split(X,Y, test_size=0.3, random_state=42)

print("X Train : ", X_train.shape)
print("X Test : ", X_test.shape)

# Random Forest
random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(X_train, Y_train)

Y_pred_rf = random_forest.predict(X_test)
random_forest_score = random_forest.score(X_test, Y_test) * 100
