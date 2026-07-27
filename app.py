import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="Bank Term Deposit Predictor", layout="centered")
st.title("Bank Term Deposit Subscription Predictor")
st.caption("Model: tuned XGBoost (holdout AUC 0.9676). Base rate in training data: ~12% subscribed overall.")

with st.form("customer_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=95, value=40)
        job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                                    'retired', 'self-employed', 'services', 'student', 'technician',
                                    'unemployed', 'unknown'])
        marital = st.selectbox("Marital status", ['divorced', 'married', 'single'])
        education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
        default = st.selectbox("Has credit in default?", ['no', 'yes'])
        balance = st.number_input("Account balance", min_value=-8020, max_value=100000, value=1200)
        housing = st.selectbox("Has housing loan?", ['no', 'yes'])
        loan = st.selectbox("Has personal loan?", ['no', 'yes'])
    with col2:
        contact = st.selectbox("Contact type", ['cellular', 'telephone', 'unknown'])
        day = st.number_input("Day of month contacted", min_value=1, max_value=31, value=15)
        month = st.selectbox("Month contacted", ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
        duration = st.number_input("Last call duration (seconds)", min_value=1, max_value=5000, value=180)
        campaign = st.number_input("Contacts this campaign", min_value=1, max_value=63, value=2)
        pdays = st.number_input("Days since last contact (-1 = never)", min_value=-1, max_value=871, value=-1)
        previous = st.number_input("Contacts before this campaign", min_value=0, max_value=200, value=0)
        poutcome = st.selectbox("Previous campaign outcome", ['failure', 'other', 'success', 'unknown'])

    submitted = st.form_submit_button("Predict")

if submitted:
    row = {'age': age, 'job': job, 'marital': marital, 'education': education, 'default': default,
           'balance': balance, 'housing': housing, 'loan': loan, 'contact': contact, 'day': day,
           'month': month, 'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome}
    row['duration'] = duration
    model = joblib.load('best_model_pipeline.joblib')
    col_order = ['age','balance','day','duration','campaign','pdays','previous',
                 'job','marital','education','default','housing','loan','contact','month','poutcome']

    X_input = pd.DataFrame([row])[col_order]
    proba = model.predict_proba(X_input)[0, 1]

    st.metric("Predicted subscription probability", f"{proba:.1%}")
    if proba >= 0.5:
        st.success("Likely to subscribe")
    else:
        st.warning("Unlikely to subscribe")

    with st.expander("What drove THIS prediction? (SHAP explanation)"):
        clf = model.named_steps['clf']
        prep = model.named_steps['prep']
        feature_names = prep.get_feature_names_out()
        Xt = prep.transform(X_input)

        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)[0]
        contrib = pd.Series(sv, index=feature_names).sort_values(key=abs, ascending=False).head(8)
        chart_contrib = contrib.iloc[::-1]

        colors = ['#1B9E77' if v > 0 else '#D95F02' for v in chart_contrib.values]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh([f.replace('num__', '').replace('cat__', '') for f in chart_contrib.index], chart_contrib.values, color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Contribution to this prediction (log-odds scale)')
        st.pyplot(fig)
        st.caption("Green = pushed toward 'subscribe', orange = pushed toward 'not subscribe'. "
                   "Specific to this one customer — unlike a global importance chart, this changes every time you predict.")

        def describe(fname, row):
            if fname.startswith('cat__'):
                base = fname.replace('cat__', '')
                for col in ['job','marital','education','default','housing','loan','contact','month','poutcome']:
                    if base.startswith(col + '_'):
                        return f'{col} = {row[col]}'
                return base
            else:
                base = fname.replace('num__', '')
                return f'{base} ({row.get(base)})'

        max_abs = contrib.abs().max()
        threshold = max_abs * 0.15
        top_pos = contrib[(contrib > 0) & (contrib.abs() >= threshold)].sort_values(ascending=False).head(3)
        top_neg = contrib[(contrib < 0) & (contrib.abs() >= threshold)].sort_values().head(3)
        pos_text = ', '.join(describe(f, row) for f in top_pos.index) if len(top_pos) else 'no strong positive factors'
        neg_text = ', '.join(describe(f, row) for f in top_neg.index) if len(top_neg) else 'no strong negative factors'

        st.markdown(f"**Prediction summary:** This customer has a **{proba:.1%}** probability of subscribing. "
                    f"Strongest factors pushing toward *subscribe*: {pos_text}. "
                    f"Strongest factors pushing toward *not subscribe*: {neg_text}.")