import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Set page config
st.set_page_config(page_title="My Model", layout="wide")

# Title and introduction
st.title("Iris Dataset Explorer")
st.write("This app demonstrates analysis of the iris dataset")

# Sidebar
st.sidebar.header("Iris")
feature = st.sidebar.radio('Select one:', ['Explore Dataset', 'K-Means Clustering'])

# Main content
if feature == 'Explore Dataset':
  st.header("Data Exploration")

  st.subheader("1. Show any 5 sample records")
  st.dataframe(iris.sample(5))

  st.subheader("2. Descriptive Statistics")
  st.table(iris.describe())

  st.subheader("3. Correlation Matrix")
  st.write(iris.drop(columns=['species']).corr())

elif feature == 'K-Means Clustering':
  st.header("K-Means Clustering")

  # Prepare data for clustering
  features = iris.columns[:-1]
  X = iris[features]
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # Allow user to choose number of clusters
  k = st.slider('Select number of clusters (k) ', 2, 10, 3)

  # Perform K-Means clustering
  kmeans = KMeans(n_clusters = k, random_state = 42)
  iris['cluster'] = kmeans.fit_predict(X_scaled)

  # Visualise cluster
  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Clusters Based on Sepal")
    fig, ax = plt.subplots()
    sns.scatterplot(x='sepal_length', y='sepal_width', hue='cluster', data=iris, ax=ax)
    st.pyplot(fig)
  with col2:
    st.subheader("Clusters Based on Petal")
    fig, ax = plt.subplots()
    sns.scatterplot(x='petal_length', y='petal_width', hue='cluster', data=iris, ax=ax)
    st.pyplot(fig)

# Sidebar footer
st.sidebar.markdown('---')
st.sidebar.write('Done by Mr. Ben on 22 Jul 2024')

# Footer
st.markdown('---')
if st.button('Project Completed!'):
  st.balloons()
