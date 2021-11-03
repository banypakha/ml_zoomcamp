import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import xgboost as xgb

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import DictVectorizer


df = pd.read_csv('aug_train.csv')


# Data Cleaning
experience_values = { '>20' : '21', '<1' : '0'}
df.experience = df.experience.replace(experience_values)
df['experience'] = pd.to_numeric(df['experience'])

last_new_job_values = {'>4' : '4+'}
df.last_new_job = df.last_new_job.replace(last_new_job_values)

company_size_values = { '10/49' : '10-49', '<10' : '0-10'}
df.company_size = df.company_size.replace(company_size_values)

# Split into Train and Testing
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
y_full_train = df_full_train.target.values
y_test = df_test.target.values


del df_test['target']
del df_test['enrollee_id']
del df_full_train['target']
del df_full_train['enrollee_id']

# Impute Missing Value
imputer = SimpleImputer(strategy='constant')
df_full_train_impute = imputer.fit_transform(df_full_train)
df_test_impute = imputer.transform(df_test)

# Convert back to DataFrame
df_full_train = pd.DataFrame(df_full_train_impute, columns = df_full_train.columns)
df_test = pd.DataFrame(df_test_impute, columns = df_full_train.columns)

# Encoding Categorical Features
dv = DictVectorizer(sparse=False)

full_train_dict = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)

# Convert to DMatrix for XGBoost
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train,
                    feature_names=dv.get_feature_names())

dtest = xgb.DMatrix(X_test, feature_names=dv.get_feature_names())

# Train the model
xgb_params = {
    'eta': 0.1, 
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dfulltrain, num_boost_round=30)


# Evaluate the Model
y_pred = model.predict(dtest)
roc_auc = roc_auc_score(y_test, y_pred)
print('roc_auc_score xgboost : ', roc_auc)

y_predict = y_pred >= 0.3
f1 = f1_score(y_test, y_predict)
print('f1_score xgboost : ', f1)

# Save the model, DictVectorizer, and Imputer
with open('model.bin', 'wb') as f_out:
   pickle.dump(model, f_out)

with open('dv.bin', 'wb') as f_out:
   pickle.dump(dv, f_out)

with open('imputer.bin', 'wb') as f_out:
   pickle.dump(imputer, f_out)