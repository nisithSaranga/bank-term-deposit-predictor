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

# %%
from sklearn.svm import SVC

X_svm_sub, _, y_svm_sub, _ = train_test_split(X, y, train_size=20000, stratify=y, random_state=RANDOM_STATE)
X_svm_tr, X_svm_val, y_svm_tr, y_svm_val = train_test_split(X_svm_sub, y_svm_sub, test_size=0.2, stratify=y_svm_sub, random_state=RANDOM_STATE)

svm_pipe = Pipeline([
    ('prep', make_preprocessor()),
    ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE))
])
svm_pipe.fit(X_svm_tr, y_svm_tr)
svm_auc = roc_auc_score(y_svm_val, svm_pipe.predict_proba(X_svm_val)[:, 1])
print(f'SVM (20,000-row subsample) holdout AUC: {svm_auc:.4f}')

# %% 
svm_pipe.fit(X_svm_sub, y_svm_sub)  # refit on the full 20k subsample, not just the 80% split
submission4 = pd.DataFrame({'id': test['id'], 'y': svm_pipe.predict_proba(X_test)[:, 1]})
submission4.to_csv('submission_4_svm.csv', index=False)
print('saved:', submission4.shape)

# %%
from sklearn.neural_network import MLPClassifier

mlp_pipe = Pipeline([
    ('prep', make_preprocessor()),
    ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=60, early_stopping=True, random_state=RANDOM_STATE))
])
mlp_pipe.fit(X_tr, y_tr)
mlp_auc = roc_auc_score(y_val, mlp_pipe.predict_proba(X_val)[:, 1])
print(f'Neural Network holdout AUC: {mlp_auc:.4f}')

# %%
mlp_pipe.fit(X, y)
submission5 = pd.DataFrame({'id': test['id'], 'y': mlp_pipe.predict_proba(X_test)[:, 1]})
submission5.to_csv('submission_5_neuralnet.csv', index=False)
print('saved:', submission5.shape)  

# %%
NUMERIC_NO_DUR = [c for c in NUMERIC if c != 'duration']

logreg_no_dur_pipe = Pipeline([
    ('prep', make_preprocessor(NUMERIC_NO_DUR)),
    ('clf', LogisticRegression(max_iter=500, class_weight='balanced', random_state=RANDOM_STATE))
])
logreg_no_dur_pipe.fit(X_tr[NUMERIC_NO_DUR + CATEGORICAL], y_tr)
no_dur_auc = roc_auc_score(y_val, logreg_no_dur_pipe.predict_proba(X_val[NUMERIC_NO_DUR + CATEGORICAL])[:, 1])
print(f'Logistic Regression WITHOUT duration, holdout AUC: {no_dur_auc:.4f}')

# %%
X_no_dur = X[NUMERIC_NO_DUR + CATEGORICAL]
X_test_no_dur = X_test[NUMERIC_NO_DUR + CATEGORICAL]

logreg_no_dur_pipe.fit(X_no_dur, y)  # refit on all of train.csv, features minus duration
submission6 = pd.DataFrame({'id': test['id'], 'y': logreg_no_dur_pipe.predict_proba(X_test_no_dur)[:, 1]})
submission6.to_csv('submission_6_logreg_no_duration.csv', index=False)
print('saved:', submission6.shape)

# %%
from xgboost import XGBClassifier
prep_fitted = make_preprocessor().fit(X_tr, y_tr)
Xt_tr = prep_fitted.transform(X_tr)
Xt_val = prep_fitted.transform(X_val)
scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

xgb_tuned = XGBClassifier(n_estimators=1000, max_depth=7, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                           eval_metric='auc', scale_pos_weight=scale_pos_weight,
                           n_jobs=-1, early_stopping_rounds=30, random_state=RANDOM_STATE)
xgb_tuned.fit(Xt_tr, y_tr, eval_set=[(Xt_val, y_val)], verbose=False)
tuned_auc = roc_auc_score(y_val, xgb_tuned.predict_proba(Xt_val)[:, 1])
print(f'Tuned XGBoost holdout AUC: {tuned_auc:.4f}  (best_iteration: {xgb_tuned.best_iteration})')

# %%
best_n = xgb_tuned.best_iteration
prep_final = make_preprocessor().fit(X, y)
Xt_all = prep_final.transform(X)
scale_pos_weight_full = (y == 0).sum() / (y == 1).sum()

xgb_final = XGBClassifier(n_estimators=best_n, max_depth=7, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                           eval_metric='auc', scale_pos_weight=scale_pos_weight_full,
                           n_jobs=-1, random_state=RANDOM_STATE)
xgb_final.fit(Xt_all, y)
best_model_pipeline = Pipeline([('prep', prep_final), ('clf', xgb_final)])

import joblib
joblib.dump(best_model_pipeline, 'best_model_pipeline.joblib')  # <- Streamlit loads this

submission7 = pd.DataFrame({'id': test['id'], 'y': best_model_pipeline.predict_proba(X_test)[:, 1]})
submission7.to_csv('submission_7_xgboost_tuned.csv', index=False)  # <- Kaggle gets this, tomorrow
print('saved model + submission:', submission7.shape)