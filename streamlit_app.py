import streamlit as st
import pandas as pd
import numpy as np
import category_encoders as ce
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="🐧 Penguin Classifier", layout="wide")
st.title('🐧 Penguin Classifier')
st.write('Working with penguin dataset')

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

st.subheader("Dataset shape")
st.write("Rows: ", df.shape[0])
st.write("Columns: ", df.shape[1])

st.subheader("🔍 Random 10 rows")
st.dataframe(df.sample(10), use_container_width=True)

st.subheader("🔍 Visualization")
col1, col2 = st.columns(2)
with col1:
  fig1 = px.histogram(df, x="species", color="island", barmode="group", title="Distribution of species across islands")
  st.plotly_chart(fig1, use_container_width=True)
with col2:
  fig2 = px.scatter(df, x="bill_length_mm", y="flipper_length_mm", color="species", title="Bill length vs Flipper length")
  st.plotly_chart(fig2, use_container_width=True)

X = df.drop(["species"], axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

encoder = ce.TargetEncoder(cols=['island', 'sex'])
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

models = {
  'Decision Tree': DecisionTreeClassifier(random_state=42),
  'KNN': KNeighborsClassifier()
}

results = []
for name, model in models.items():
  model.fit(X_train_encoded, y_train)
  acc_train = accuracy_score(y_train, model.predict(X_train_encoded))
  acc_test = accuracy_score(y_test, model.predict(X_test_encoded))
  results.append({
    'Model': name,
    'Train Accuracy': acc_train,
    'Test Accuracy': acc_test
  })

st.subheader("Comparing models metrics")
st.table(pd.Dataframe(results))
