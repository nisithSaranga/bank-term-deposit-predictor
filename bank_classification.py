# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

RANDOM_STATE = 42
sns.set_style('whitegrid')

# %%
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')
print('train:', train.shape, ' test:', test.shape)
train.head()

# %%
print('Missing values — train:', train.isnull().sum().sum(), ' test:', test.isnull().sum().sum())
print('Duplicate rows in train:', train.duplicated().sum())

# %%
NUMERIC = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATEGORICAL = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

print(train['y'].value_counts(normalize=True).round(4))

# %%
print(train.groupby('y')['duration'].mean().round(1))

# %%
X = train[NUMERIC + CATEGORICAL]
y = train['y']
X_test = test[NUMERIC + CATEGORICAL]

def make_preprocessor(numeric_cols=NUMERIC):
    return ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CATEGORICAL)
    ])

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(X_tr.shape, X_val.shape)

# %%
logreg_pipe = Pipeline([
    ('prep', make_preprocessor()),
    ('clf', LogisticRegression(max_iter=500, class_weight='balanced', random_state=RANDOM_STATE))
])
logreg_pipe.fit(X_tr, y_tr)
val_proba = logreg_pipe.predict_proba(X_val)[:, 1]
logreg_auc = roc_auc_score(y_val, val_proba)
print(f'Logistic Regression holdout AUC: {logreg_auc:.4f}')

# %%
NUMERIC_NO_DUR = [c for c in NUMERIC if c != 'duration']
logreg_no_dur = Pipeline([
    ('prep', make_preprocessor(NUMERIC_NO_DUR)),
    ('clf', LogisticRegression(max_iter=500, class_weight='balanced', random_state=RANDOM_STATE))
])
logreg_no_dur.fit(X_tr[NUMERIC_NO_DUR + CATEGORICAL], y_tr)
auc_no_dur = roc_auc_score(y_val, logreg_no_dur.predict_proba(X_val[NUMERIC_NO_DUR + CATEGORICAL])[:, 1])
print(f'With duration: {logreg_auc:.4f}   Without: {auc_no_dur:.4f}   Drop: {logreg_auc - auc_no_dur:.4f}')

# %%
logreg_pipe.fit(X, y)  # refit on ALL of train.csv
submission = pd.DataFrame({'id': test['id'], 'y': logreg_pipe.predict_proba(X_test)[:, 1]})
assert list(submission.columns) == list(sample_submission.columns)
assert (submission['id'].values == sample_submission['id'].values).all()
submission.to_csv('submission_1_logreg.csv', index=False)
print('saved:', submission.shape)

# %%
from sklearn.ensemble import RandomForestClassifier

rf_pipe = Pipeline([
    ('prep', make_preprocessor()),
    ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, class_weight='balanced', random_state=RANDOM_STATE))
])
rf_pipe.fit(X_tr, y_tr)
rf_auc = roc_auc_score(y_val, rf_pipe.predict_proba(X_val)[:, 1])
print(f'Random Forest holdout AUC: {rf_auc:.4f}')

# %%
rf_pipe.fit(X, y)
submission2 = pd.DataFrame({'id': test['id'], 'y': rf_pipe.predict_proba(X_test)[:, 1]})
submission2.to_csv('submission_2_randomforest.csv', index=False)
print('saved:', submission2.shape)

# %%
from xgboost import XGBClassifier

scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()  # XGBoost's version of class_weight='balanced'

xgb_pipe = Pipeline([
    ('prep', make_preprocessor()),
    ('clf', XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric='auc',
                           scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=RANDOM_STATE))
])
xgb_pipe.fit(X_tr, y_tr)
xgb_auc = roc_auc_score(y_val, xgb_pipe.predict_proba(X_val)[:, 1])
print(f'XGBoost holdout AUC: {xgb_auc:.4f}')

# %%
xgb_pipe.fit(X, y)
submission3 = pd.DataFrame({'id': test['id'], 'y': xgb_pipe.predict_proba(X_test)[:, 1]})
submission3.to_csv('submission_3_xgboost.csv', index=False)
print('saved:', submission3.shape)

# %%
print(f"Logistic Regression : {logreg_auc:.4f}")
print(f"Random Forest        : {rf_auc:.4f}")
print(f"XGBoost               : {xgb_auc:.4f}")