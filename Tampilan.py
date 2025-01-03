import pickle
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB


data = pd.read_csv('amazon.csv')

menu = ["Show Dataset", "K-Means Clustering", "Naive Bayes","EDA", "K-Means & Sentiment Analysis"]
choice = st.sidebar.selectbox("Pilih Dashboard ", menu)

if choice == "Show Dataset":
    st.title("Amazon Data Sales")
    st.write(data)
elif choice == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Descriptive Statistics")
    st.write(data.describe())

    st.subheader("Missing Values")
    missing_values = data.isnull().sum()
    st.write(missing_values[missing_values > 0])

    st.subheader("Data Types")
    st.write(data.dtypes)

    st.subheader("Top 10 Main Categories")
# Check if the 'category' column exists in the dataset
    if 'category' in data.columns:
        # Extract the first category from the 'category' column
        main_categories = data['category'].str.split('|').str[0]
        
        # Count the top 10 main categories
        top_categories = main_categories.value_counts().head(10)

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        top_categories.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Top 10 Main Categories')
        ax.set_xticklabels(top_categories.index, rotation=45, ha='right')
        plt.tight_layout()

        # Display the chart in Streamlit
        st.pyplot(fig)
    else:
        st.warning("The column 'category' does not exist in the dataset.")

    st.subheader("Top Frequent Categories in Categorical Columns")
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        st.write(f"#### {column}")
        st.write(data[column].value_counts().head(10))
elif choice == "K-Means Clustering":
    st.title("K-Means Clustering")
    
elif choice == "K-Means Clustering":
    st.title("K-Means Clustering")

    data['actual_price'] = data['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    data['rating_count'] = data['rating_count'].str.replace(',', '').astype(float)
    selected_data = data[['product_id', 'actual_price', 'rating_count']].dropna()

    scaler = MinMaxScaler()
    selected_data[['actual_price', 'rating_count']] = scaler.fit_transform(selected_data[['actual_price', 'rating_count']])
    st.header("Isi Dataset")
    st.write(selected_data)

    inertia = []
    k_values = range(1, 11)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(selected_data[['actual_price', 'rating_count']])
        inertia.append(kmeans.inertia_)

    st.subheader("Visualisasi Metode Elbow")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, inertia, marker='o', linestyle='--', color='blue')
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Optimal K')
    ax.grid(True)
    st.pyplot(fig)
    
    st.subheader("Select Columns for Clustering")
    available_columns = data.columns.tolist()
    selected_columns = st.multiselect("Kolom:", available_columns, default=["actual_price", "rating_count"])

    selected_data = data[selected_columns].dropna()

    scaler = MinMaxScaler()
    selected_data[selected_columns] = scaler.fit_transform(selected_data[selected_columns])

    st.subheader("Define Number of Clusters (K)")
    define_k = st.slider("Jumlah klaster (K)", min_value=2, max_value=10, value=3)

    kmeans = KMeans(n_clusters=define_k, random_state=42)
    selected_data['cluster'] = kmeans.fit_predict(selected_data[selected_columns])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(selected_data[selected_columns])
    selected_data['pca1'] = pca_result[:, 0]
    selected_data['pca2'] = pca_result[:, 1]

    silhouette_avg = silhouette_score(selected_data[selected_columns], selected_data['cluster'])
    st.subheader("Silhouette Score")
    st.write(f"**Silhouette Score:** {silhouette_avg:.2f}")

    st.subheader("Visualisasi klastering dengan PCA dan Centroids")
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = sns.color_palette("tab10", define_k)
    for cluster in range(define_k):
        cluster_data = selected_data[selected_data['cluster'] == cluster]
        ax.scatter(cluster_data['pca1'], cluster_data['pca2'], label=f'Cluster {cluster}', alpha=0.6, color=colors[cluster])

    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=200, c='black', marker='X', label='Centroids')

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title(f'Clustering Visualization with PCA\n')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    if not selected_data.empty:
        st.subheader("Preview Dataset")
        st.dataframe(selected_data.style.format(precision=2))  
    else:
        st.warning("Dataset is empty. Please check the selected columns or data preprocessing steps.")

elif choice == "Naive Bayes":
    st.title("Sentimen Analisis")

    data['text_cleaning'] = data['review_content'].str.lower()
    data['text_cleaning'] = data['text_cleaning'].apply(lambda x: re.sub(r'http\S+|www.\S+', '', x))
    data['text_cleaning'] = data['text_cleaning'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    data['text_cleaning'] = data['text_cleaning'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
    data['sentiment'] = data['rating'].apply(lambda x: 'positive' if x >= 4.0 else 'negative')
    
    st.header("Sentiment Data")
    data[['review_content', 'text_cleaning', 'sentiment']]

    data_cleaned = data.replace({'positive': 1, 'negative': 0})

    st.subheader("Sentiment Sebelum Oversampling")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x=data_cleaned['sentiment'], palette=['red', 'green'], ax=ax)
    ax.set_xlabel('Sentimen')
    ax.set_ylabel('Jumlah')
    st.pyplot(fig)

    X = data_cleaned['text_cleaning']
    y = data_cleaned['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)

    sentimen_counts = y_train_resampled.value_counts()
    st.subheader("Sentiment Sesudah Oversampling")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(sentimen_counts.index, sentimen_counts.values, color=['green', 'red'])
    ax.set_xlabel('Sentimen')
    ax.set_ylabel('Jumlah')
    ax.set_xticks(sentimen_counts.index)
    ax.set_xticklabels([1, 0])
    st.pyplot(fig)

    model = MultinomialNB()
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)

    st.write(f"**Accuracy:** {acc:.2f}")

    data['sentiment_pred'] = model.predict(vectorizer.transform(data['text_cleaning']))

elif choice == "K-Means & Sentiment Analysis":
    st.title("Hubungan antara klaster dan sentimen")

    define_k = st.sidebar.slider("Jumlah Klaster (K)", min_value=2, max_value=10, value=3)
    
    st.subheader("K-Means Clustering")

    data['actual_price'] = data['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    data['rating_count'] = data['rating_count'].str.replace(',', '').astype(float)
    selected_data = data[['product_id', 'actual_price', 'rating_count']].dropna()

    scaler = MinMaxScaler()
    selected_data[['actual_price', 'rating_count']] = scaler.fit_transform(selected_data[['actual_price', 'rating_count']])

    kmeans = KMeans(n_clusters=define_k, random_state=42)
    selected_data['cluster'] = kmeans.fit_predict(selected_data[['actual_price', 'rating_count']])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(selected_data[['actual_price', 'rating_count']])
    selected_data['pca1'] = pca_result[:, 0]
    selected_data['pca2'] = pca_result[:, 1]

    silhouette_avg = silhouette_score(selected_data[['actual_price', 'rating_count']], selected_data['cluster'])
    st.subheader("Silhouette Score")
    st.write(f"**Silhouette Score:** {silhouette_avg:.2f}")

    st.subheader("Visualisasi klastering dengan PCA dan Centroids")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("tab10", define_k)
    for cluster in range(define_k):
        cluster_data = selected_data[selected_data['cluster'] == cluster]
        ax.scatter(cluster_data['pca1'], cluster_data['pca2'], label=f'Cluster {cluster}', alpha=0.6, color=colors[cluster])

    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=200, c='black', marker='X', label='Centroids')

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title(f'Clustering Visualization with PCA\n')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


    data['text_cleaning'] = data['review_content'].str.lower()
    data['text_cleaning'] = data['text_cleaning'].apply(lambda x: re.sub(r'http\S+|www.\S+', '', x))
    data['text_cleaning'] = data['text_cleaning'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    data['text_cleaning'] = data['text_cleaning'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
    data['sentiment'] = data['rating'].apply(lambda x: 'positive' if x >= 4.0 else 'negative')
    data_cleaned = data.replace({'positive': 1, 'negative': 0})

    vectorizer = CountVectorizer()
    model = MultinomialNB()

    X = data_cleaned['text_cleaning']
    y = data_cleaned['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model.fit(X_train_vec, y_train)

    data['sentiment_pred'] = model.predict(vectorizer.transform(data['text_cleaning']))

    combined_data = pd.merge(data, selected_data[['product_id', 'cluster']], on='product_id', how='left')

    st.subheader("Sentimen berdasarkan klastering")
    st.dataframe(combined_data[['product_id', 'actual_price', 'rating_count', 'cluster', 'sentiment_pred']])

    cluster_sentiment = combined_data.groupby(['cluster', 'sentiment_pred']).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_sentiment.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    ax.set_title("Sentiment berdasarkan cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Jumlah")
    ax.legend(title="Sentiment", labels=["Negative", "Positive"])
    st.pyplot(fig)

