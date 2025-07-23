import streamlit as st
import pandas as pd
import numpy as np
import category_encoders as ce
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="🐧 Penguin Classifier", layout="wide")
st.title('Penguin Classifier')
st.write('Working with penguin dataset')

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

st.subheader("Random 10 rows")
st.dataframe(df.sample(10), use_container_width=True)
