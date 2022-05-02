import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate
from helpers_tr.eda import *
from helpers_tr.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#############################################
# Feature Engineering (Değişken Mühendisliği)
# Below pickle is outcome of feature engineering implemented on titanic data. Check titanic_data_prep.py
# 1. Exploratory Data Analysis
# 2. Data Preprocessing
#############################################



df_prepared = pd.read_pickle("hafta06/df_prepared.pkl")

df_prepared.head()

check_df(df_prepared)

#############################################
# 2. Model - Random Forest Classifier
#############################################

y = df_prepared["SURVIVED"]
X = df_prepared.drop(["PASSENGERID", "SURVIVED"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# accuracy score: 0.8097014925373134

#############################################
# 2. Model - Logistic Regression
#############################################

# Bağımlı ve bağımsız değişkelerin seçilmesi:
y = df_prepared["SURVIVED"]
X = df_prepared.drop(["PASSENGERID", "SURVIVED"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.30, random_state=17)

# Model:
log_model = LogisticRegression().fit(X_train, y_train)

log_model.intercept_
log_model.coef_


# Tahmin
y_pred = log_model.predict(X)

y_pred[0:10]
y[0:10]

# Test setinin modele sorulması:
y_pred = log_model.predict(X_test)

# AUC Score için y_prob (1. sınıfa ait olma olasılıkları)
y_prob = log_model.predict_proba(X_test)[:, 1]

# ROC Curve
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

roc_auc_score(y_test, y_prob)

# accuracy score: 0.8497159579962127