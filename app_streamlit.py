import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===== Manual Label Mapping =====
index_to_label = {0: 'positif', 1: 'negatif', 2: 'netral'}
label_to_index = {v: k for k, v in index_to_label.items()}

# ===== Load All Models & Data =====
instagram_model_unbalanced = joblib.load('models/unbalancing/instagram_model.joblib')
instagram_vectorizer_unbalanced = joblib.load('models/unbalancing/instagram_vectorizer.joblib')
instagram_selector_unbalanced = joblib.load('models/unbalancing/instagram_selector.joblib')

twitter_model_unbalanced = joblib.load('models/unbalancing/twitter_model.joblib')
twitter_vectorizer_unbalanced = joblib.load('models/unbalancing/twitter_vectorizer.joblib')
twitter_selector_unbalanced = joblib.load('models/unbalancing/twitter_selector.joblib')

instagram_model_balanced = joblib.load('models/balancing/instagram_model.joblib')
instagram_vectorizer_balanced = joblib.load('models/balancing/instagram_vectorizer.joblib')
instagram_selector_balanced = joblib.load('models/balancing/instagram_selector.joblib')

twitter_model_balanced = joblib.load('models/balancing/twitter_model.joblib')
twitter_vectorizer_balanced = joblib.load('models/balancing/twitter_vectorizer.joblib')
twitter_selector_balanced = joblib.load('models/balancing/twitter_selector.joblib')

# ===== Prediction Function =====
def predict_sentiment(comment, model, vectorizer, selector):
    vect = vectorizer.transform([comment])
    selected = selector.transform(vect)
    pred = model.predict(selected)
    return index_to_label.get(pred[0], 'unknown')  # Convert angka ke label

# ===== Sidebar =====
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["Home", "Prediction", "Accuracy", "Visualization", "Preprocessing"])

# ===== Home =====
if menu == "Home":
    st.title("Sentiment Analysis Dashboard")
    st.markdown("Aplikasi Analisis Sentimen untuk Instagram & Twitter.")

# ===== Prediction Page =====
elif menu == "Prediction":
    st.title("Predict Sentiment")

    platform = st.selectbox("Platform", ["Instagram", "Twitter"])
    balanced = st.checkbox("Use Balanced Model")
    comment = st.text_area("Enter Comment")

    if st.button("Predict"):
        if not comment.strip():
            st.error("Comment cannot be empty!")
        else:
            if platform == "Instagram":
                if balanced:
                    result = predict_sentiment(comment, instagram_model_balanced, instagram_vectorizer_balanced, instagram_selector_balanced)
                else:
                    result = predict_sentiment(comment, instagram_model_unbalanced, instagram_vectorizer_unbalanced, instagram_selector_unbalanced)
                st.success(f"Predicted Sentiment: {result}")
            elif platform == "Twitter":
                if balanced:
                    result = predict_sentiment(comment, twitter_model_balanced, twitter_vectorizer_balanced, twitter_selector_balanced)
                else:
                    result = predict_sentiment(comment, twitter_model_unbalanced, twitter_vectorizer_unbalanced, twitter_selector_unbalanced)
                st.success(f"Predicted Sentiment: {result}")

# ===== Accuracy Page =====
elif menu == "Accuracy":
    st.title("Model Accuracy")

    df_test = pd.read_csv('data/unbalancing/instagram/testing_data.csv')
    vect = instagram_vectorizer_unbalanced.transform(df_test['preprocessing_comment'])
    selected = instagram_selector_unbalanced.transform(vect)
    preds_num = instagram_model_unbalanced.predict(selected)
    preds = [index_to_label.get(p, 'unknown') for p in preds_num]  # Convert angka ke label

    acc = accuracy_score(df_test['Sentiment'], preds)
    st.write(f"Accuracy: {acc * 100:.2f}%")

    st.text("Classification Report")
    st.text(classification_report(df_test['Sentiment'], preds))

    cm = confusion_matrix(df_test['Sentiment'], preds, labels=['positif', 'negatif', 'netral'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['positif', 'negatif', 'netral'], yticklabels=['positif', 'negatif', 'netral'])
    st.pyplot(fig)

# ===== Visualization Page =====
elif menu == "Visualization":
    st.title("Visualization")

    try:
        df = pd.read_csv('dataset/label_ig.csv')
        sentiment_count = df['Sentiment'].value_counts()

        fig, ax = plt.subplots()
        sns.barplot(x=sentiment_count.index, y=sentiment_count.values, palette='viridis', ax=ax)
        ax.set_title("Sentiment Distribution")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.subheader("Word Cloud")
        text = ' '.join(df['Cleaned Comment'].astype(str))
        wordcloud = WordCloud(width=800, height=400).generate(text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    except Exception as e:
        st.error(f"Error loading data: {e}")

# ===== Preprocessing Page =====
elif menu == "Preprocessing":
    st.title("Preprocessing")

    process_type = st.selectbox("Choose Process", ["Normalisasi", "Tokenisasi", "Stopword", "Stemming"])

    if process_type:
        try:
            df_pre = pd.read_csv(f'dataset/olahdata/preprocessing_ig_{process_type.lower()}.csv')
            st.dataframe(df_pre.head(20))
        except FileNotFoundError:
            st.error("Data not found for selected process type.")
        except Exception as e:
            st.error(f"Error loading data: {e}")
