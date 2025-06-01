import joblib
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import sent_tokenize
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from naive_bayes import NaiveBayes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np

app = Flask(__name__)

# Load models, vectorizers, and selectors for both balanced and unbalanced
instagram_model_unbalanced = joblib.load('models/unbalancing/Instagram_model (1).joblib')
instagram_vectorizer_unbalanced = joblib.load('models/unbalancing/Instagram_vectorizer (1).joblib')
instagram_selector_unbalanced = joblib.load('models/unbalancing/Instagram_selector (1).joblib')

instagram_model_balanced = joblib.load('models/balancing/Instagram_model (2).joblib')
instagram_vectorizer_balanced = joblib.load('models/balancing/Instagram_vectorizer (2).joblib')
instagram_selector_balanced = joblib.load('models/balancing/Instagram_selector (2).joblib')

twitter_model_unbalanced = joblib.load('models/unbalancing/Twitter_model (1).joblib')
twitter_vectorizer_unbalanced = joblib.load('models/unbalancing/Twitter_vectorizer (1).joblib')
twitter_selector_unbalanced = joblib.load('models/unbalancing/Twitter_selector (1).joblib')

twitter_model_balanced = joblib.load('models/balancing/Twitter_model (2).joblib')
twitter_vectorizer_balanced = joblib.load('models/balancing/Twitter_vectorizer (2).joblib')
twitter_selector_balanced = joblib.load('models/balancing/Twitter_selector (2).joblib')

# Function to preprocess and predict sentiment for Instagram
def predict_instagram_comment(comment, balanced=False):
    if balanced:
        comment_vect = instagram_vectorizer_balanced.transform([comment])
        comment_selected = instagram_selector_balanced.transform(comment_vect)
        prediction = instagram_model_balanced.predict(comment_selected)
    else:
        comment_vect = instagram_vectorizer_unbalanced.transform([comment])
        comment_selected = instagram_selector_unbalanced.transform(comment_vect)
        prediction = instagram_model_unbalanced.predict(comment_selected)
    
    return prediction[0]

# Function to preprocess and predict sentiment for Twitter
def predict_twitter_comment(comment, balanced=False):
    if balanced:
        comment_vect = twitter_vectorizer_balanced.transform([comment])
        comment_selected = twitter_selector_balanced.transform(comment_vect)
        prediction = twitter_model_balanced.predict(comment_selected)
    else:
        comment_vect = twitter_vectorizer_unbalanced.transform([comment])
        comment_selected = twitter_selector_unbalanced.transform(comment_vect)
        prediction = twitter_model_unbalanced.predict(comment_selected)
    
    return prediction[0]

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        comment = request.form['comment']
        platform = request.form['platform']
        balanced = 'balanced' in request.form

        # Cek apakah komentar kosong
        if not comment:
            error_message = "Komentar tidak boleh kosong!"
            return render_template('predict.html', error_message=error_message)
        
        if platform == 'Instagram':
            sentiment = predict_instagram_comment(comment, balanced)
        else:
            sentiment = predict_twitter_comment(comment, balanced)
        
        return render_template('predict.html', result=sentiment)

    return render_template('predict.html')


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

@app.route('/accuracy')
def accuracy():
    # Load the testing data for both balanced and unbalanced
    instagram_test_balanced = pd.read_csv('data/balancing/instagram/testing_data.csv')
    instagram_train_balanced = pd.read_csv('data/balancing/instagram/training_data.csv')

    instagram_test_unbalanced = pd.read_csv('data/unbalancing/instagram/testing_data.csv')
    instagram_train_unbalanced = pd.read_csv('data/unbalancing/instagram/training_data.csv')

    twitter_test_balanced = pd.read_csv('data/balancing/twitter/testing_data.csv')
    twitter_train_balanced = pd.read_csv('data/balancing/twitter/training_data.csv')

    twitter_test_unbalanced = pd.read_csv('data/unbalancing/twitter/testing_data.csv')
    twitter_train_unbalanced = pd.read_csv('data/unbalancing/twitter/training_data.csv')

    # Instagram Metrics (Unbalanced)
    instagram_test_vect_unbalanced = instagram_vectorizer_unbalanced.transform(instagram_test_unbalanced['preprocessing_comment'])
    instagram_test_selected_unbalanced = instagram_selector_unbalanced.transform(instagram_test_vect_unbalanced)
    instagram_predictions_unbalanced = instagram_model_unbalanced.predict(instagram_test_selected_unbalanced)

    # Calculate Metrics for Instagram (Unbalanced)
    instagram_accuracy_unbalanced = accuracy_score(instagram_test_unbalanced['Sentiment'], instagram_predictions_unbalanced)
    instagram_precision_unbalanced = precision_score(instagram_test_unbalanced['Sentiment'], instagram_predictions_unbalanced, average='weighted', labels=['positif', 'negatif', 'netral'])
    instagram_recall_unbalanced = recall_score(instagram_test_unbalanced['Sentiment'], instagram_predictions_unbalanced, average='weighted', labels=['positif', 'negatif', 'netral'])
    instagram_f1_unbalanced = f1_score(instagram_test_unbalanced['Sentiment'], instagram_predictions_unbalanced, average='weighted', labels=['positif', 'negatif', 'netral'])

    # Instagram Metrics (Balanced)
    instagram_test_vect_balanced = instagram_vectorizer_balanced.transform(instagram_test_balanced['preprocessing_comment'])
    instagram_test_selected_balanced = instagram_selector_balanced.transform(instagram_test_vect_balanced)
    instagram_predictions_balanced = instagram_model_balanced.predict(instagram_test_selected_balanced)

    # Calculate Metrics for Instagram (Balanced)
    instagram_accuracy_balanced = accuracy_score(instagram_test_balanced['Sentiment'], instagram_predictions_balanced)
    instagram_precision_balanced = precision_score(instagram_test_balanced['Sentiment'], instagram_predictions_balanced, average='weighted', labels=['positif', 'negatif', 'netral'])
    instagram_recall_balanced = recall_score(instagram_test_balanced['Sentiment'], instagram_predictions_balanced, average='weighted', labels=['positif', 'negatif', 'netral'])
    instagram_f1_balanced = f1_score(instagram_test_balanced['Sentiment'], instagram_predictions_balanced, average='weighted', labels=['positif', 'negatif', 'netral'])

    # Twitter Metrics (Unbalanced)
    twitter_test_vect_unbalanced = twitter_vectorizer_unbalanced.transform(twitter_test_unbalanced['preprocessing_comment'])
    twitter_test_selected_unbalanced = twitter_selector_unbalanced.transform(twitter_test_vect_unbalanced)
    twitter_predictions_unbalanced = twitter_model_unbalanced.predict(twitter_test_selected_unbalanced)

    # Calculate Metrics for Twitter (Unbalanced)
    twitter_accuracy_unbalanced = accuracy_score(twitter_test_unbalanced['Sentiment'], twitter_predictions_unbalanced)
    twitter_precision_unbalanced = precision_score(twitter_test_unbalanced['Sentiment'], twitter_predictions_unbalanced, average='weighted', labels=['positif', 'negatif', 'netral'])
    twitter_recall_unbalanced = recall_score(twitter_test_unbalanced['Sentiment'], twitter_predictions_unbalanced, average='weighted', labels=['positif', 'negatif', 'netral'])
    twitter_f1_unbalanced = f1_score(twitter_test_unbalanced['Sentiment'], twitter_predictions_unbalanced, average='weighted', labels=['positif', 'negatif', 'netral'])

    # Twitter Metrics (Balanced)
    twitter_test_vect_balanced = twitter_vectorizer_balanced.transform(twitter_test_balanced['preprocessing_comment'])
    twitter_test_selected_balanced = twitter_selector_balanced.transform(twitter_test_vect_balanced)
    twitter_predictions_balanced = twitter_model_balanced.predict(twitter_test_selected_balanced)

    # Calculate Metrics for Twitter (Balanced)
    twitter_accuracy_balanced = accuracy_score(twitter_test_balanced['Sentiment'], twitter_predictions_balanced)
    twitter_precision_balanced = precision_score(twitter_test_balanced['Sentiment'], twitter_predictions_balanced, average='weighted', labels=['positif', 'negatif', 'netral'])
    twitter_recall_balanced = recall_score(twitter_test_balanced['Sentiment'], twitter_predictions_balanced, average='weighted', labels=['positif', 'negatif', 'netral'])
    twitter_f1_balanced = f1_score(twitter_test_balanced['Sentiment'], twitter_predictions_balanced, average='weighted', labels=['positif', 'negatif', 'netral'])

    # Confusion Matrices
    instagram_conf_matrix_unbalanced = confusion_matrix(instagram_test_unbalanced['Sentiment'], instagram_predictions_unbalanced, labels=['positif', 'negatif', 'netral'])
    instagram_conf_matrix_balanced = confusion_matrix(instagram_test_balanced['Sentiment'], instagram_predictions_balanced, labels=['positif', 'negatif', 'netral'])
    twitter_conf_matrix_unbalanced = confusion_matrix(twitter_test_unbalanced['Sentiment'], twitter_predictions_unbalanced, labels=['positif', 'negatif', 'netral'])
    twitter_conf_matrix_balanced = confusion_matrix(twitter_test_balanced['Sentiment'], twitter_predictions_balanced, labels=['positif', 'negatif', 'netral'])

    # Calculate manual metrics per class based on confusion matrices
    # For Instagram Unbalanced
    instagram_manual_metrics_unbalanced = calculate_metrics_from_conf_matrix(instagram_conf_matrix_unbalanced)
    
    # For Instagram Balanced
    instagram_manual_metrics_balanced = calculate_metrics_from_conf_matrix(instagram_conf_matrix_balanced)
    
    # For Twitter Unbalanced
    twitter_manual_metrics_unbalanced = calculate_metrics_from_conf_matrix(twitter_conf_matrix_unbalanced)
    
    # For Twitter Balanced
    twitter_manual_metrics_balanced = calculate_metrics_from_conf_matrix(twitter_conf_matrix_balanced)

    return render_template(
        'accuracy.html',
        # Accuracy values
        instagram_accuracy_unbalanced=instagram_accuracy_unbalanced * 100,  # Convert to percentage
        instagram_accuracy_balanced=instagram_accuracy_balanced * 100,
        twitter_accuracy_unbalanced=twitter_accuracy_unbalanced * 100,
        twitter_accuracy_balanced=twitter_accuracy_balanced * 100,
        # Precision values
        instagram_precision_unbalanced=instagram_precision_unbalanced,
        instagram_precision_balanced=instagram_precision_balanced,
        twitter_precision_unbalanced=twitter_precision_unbalanced,
        twitter_precision_balanced=twitter_precision_balanced,
        # Recall values
        instagram_recall_unbalanced=instagram_recall_unbalanced,
        instagram_recall_balanced=instagram_recall_balanced,
        twitter_recall_unbalanced=twitter_recall_unbalanced,
        twitter_recall_balanced=twitter_recall_balanced,
        # F1 values
        instagram_f1_unbalanced=instagram_f1_unbalanced,
        instagram_f1_balanced=instagram_f1_balanced,
        twitter_f1_unbalanced=twitter_f1_unbalanced,
        twitter_f1_balanced=twitter_f1_balanced,
        # Confusion Matrices
        instagram_conf_matrix_unbalanced=instagram_conf_matrix_unbalanced,
        instagram_conf_matrix_balanced=instagram_conf_matrix_balanced,
        twitter_conf_matrix_unbalanced=twitter_conf_matrix_unbalanced,
        twitter_conf_matrix_balanced=twitter_conf_matrix_balanced,
        # Manual metrics per class
        instagram_manual_metrics_unbalanced=instagram_manual_metrics_unbalanced,
        instagram_manual_metrics_balanced=instagram_manual_metrics_balanced,
        twitter_manual_metrics_unbalanced=twitter_manual_metrics_unbalanced,
        twitter_manual_metrics_balanced=twitter_manual_metrics_balanced
    )

def calculate_metrics_from_conf_matrix(conf_matrix):
    """
    Calculate precision, recall, and F1-score for each class from confusion matrix
    
    Args:
        conf_matrix: Confusion matrix in format [positif, negatif, netral]
    
    Returns:
        Dictionary with metrics for each class
    """
    classes = ['positif', 'negatif', 'netral']
    metrics = {}
    
    for i, class_name in enumerate(classes):
        # True Positives: diagonal element for this class
        tp = conf_matrix[i, i]
        
        # False Positives: sum of column i except diagonal element
        fp = sum(conf_matrix[:, i]) - tp
        
        # False Negatives: sum of row i except diagonal element
        fn = sum(conf_matrix[i, :]) - tp
        
        # True Negatives: sum of all elements except row i and column i
        tn = conf_matrix.sum() - (tp + fp + fn)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),  # Convert to int for better display in template
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
    
    return metrics


    
# Function to plot and display sentiment distribution
def plot_sentiment_distribution(sentiment_count, title, ax):
    ax.bar(sentiment_count.index, sentiment_count.values, color='skyblue')
    ax.set_title(title)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')

    # Add numbers on top of the bars
    for i, v in enumerate(sentiment_count.values):
        ax.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=12, color='black')

@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    result = None  # Initialize result variable
    error = None  # Initialize error variable
    page = request.args.get('page', 1, type=int)  # Get page number from query string, default 1

    # Default values for platform, process_type, and per_page
    platform = request.args.get('platform', 'Instagram')  # Default to 'Instagram'
    process_type = request.args.get('process_type', 'Normalisasi')  # Default to 'Normalisasi'
    per_page = int(request.args.get('per_page', 10))  # Default to 10 per page

    # Initialize pagination as a safe empty dict
    pagination = {
        'total': 0,
        'per_page': per_page,
        'page': page,
        'pages': 1
    }

    if request.method == 'POST':
        # Get form values
        platform = request.form.get('platform', platform)  # Use default if not set
        process_type = request.form.get('process_type', process_type)  # Use default if not set
        per_page = int(request.form.get('per_page', per_page))  # Use default if not set

        # Check if platform or process_type are missing
        if not platform or not process_type:
            error = "Platform or process type is missing!"  # Set error message
            return render_template('preprocessing.html', error=error)

        try:
            # Load the corresponding data based on platform and process type
            if platform == 'Instagram':
                if process_type == 'Normalisasi':
                    data = pd.read_csv('dataset/olahdata/preprocessing_ig_normalisasi.csv')
                elif process_type == 'Tokenisasi':
                    data = pd.read_csv('dataset/olahdata/preprocessing_ig_tokenisasi.csv')
                elif process_type == 'Stopword':
                    data = pd.read_csv('dataset/olahdata/preprocessing_ig_stopword.csv')
                elif process_type == 'Stemming':
                    data = pd.read_csv('dataset/olahdata/preprocessing_ig_stemming.csv')

            elif platform == 'Twitter':
                if process_type == 'Normalisasi':
                    data = pd.read_csv('dataset/olahdata/preprocessing_twitter_normalisasi.csv')
                elif process_type == 'Tokenisasi':
                    data = pd.read_csv('dataset/olahdata/preprocessing_twitter_tokenisasi.csv')
                elif process_type == 'Stopword':
                    data = pd.read_csv('dataset/olahdata/preprocessing_twitter_stopword.csv')
                elif process_type == 'Stemming':
                    data = pd.read_csv('dataset/olahdata/preprocessing_twitter_stemming.csv')

            # Check if data is loaded and contains the expected columns
            if data.empty:
                error = "No data found in the selected file."
                return render_template('preprocessing.html', error=error)

            before_column = 'before_' + process_type.lower()
            after_column = 'after_' + process_type.lower()

            # Check if the columns exist in the data
            if before_column not in data.columns or after_column not in data.columns:
                error = f"Columns {before_column} or {after_column} not found in the data."
                return render_template('preprocessing.html', error=error)

            # Pagination logic: handle the data per page
            total_data = len(data)
            before_data = data[before_column].iloc[(page-1)*per_page:page*per_page]
            after_data = data[after_column].iloc[(page-1)*per_page:page*per_page]

            # Update pagination dictionary
            pagination = {
                'total': total_data,
                'per_page': per_page,
                'page': page,
                'pages': (total_data // per_page) + (1 if total_data % per_page > 0 else 0)
            }

            # Assign result
            result = {
                'before': before_data,
                'after': after_data,
            }

        except Exception as e:
            error = f"An error occurred while processing data: {str(e)}"  # Handle any data loading or processing errors
            return render_template('preprocessing.html', error=error)

    return render_template('preprocessing.html', result=result, error=error, platform=platform, process_type=process_type, per_page=per_page, page=page, pagination=pagination)

# Function to generate sentiment distribution bar chart
def generate_sentiment_distribution(data):
    sentiment_counts = data['Sentiment'].value_counts()

    # Create the plot
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax, color=['#ff6666', '#66b3ff', '#99ff99'])
    ax.set_title('Sentiment Distribution')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Encode the image as base64 for embedding in the HTML
    return base64.b64encode(img.getvalue()).decode()

# Function to generate WordCloud
def generate_wordcloud(data, sentiment_type):
    sentiment_data = data[data['Sentiment'] == sentiment_type]['Cleaned Comment']
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(sentiment_data))
    img_wc = io.BytesIO()
    wordcloud.to_image().save(img_wc, format='PNG')
    img_wc.seek(0)
    return base64.b64encode(img_wc.getvalue()).decode()

# Function to generate WordCloud for Total Sentiment
def generate_total_wordcloud(data):
    # Combine all sentiment types (positive, negative, and neutral) into a single string
    total_sentiment_data = " ".join(data['Cleaned Comment'])  # Join all cleaned comments
    wordcloud = WordCloud(width=800, height=400).generate(total_sentiment_data)
    img_wc = io.BytesIO()
    wordcloud.to_image().save(img_wc, format='PNG')
    img_wc.seek(0)
    return base64.b64encode(img_wc.getvalue()).decode()

@app.route('/visualization')
def visualization():
    # Load data
    instagram_data_unbalanced = pd.read_csv('dataset/before_softbal_label_ig.csv')
    instagram_data_balanced = pd.read_csv('dataset/label_ig.csv')
    twitter_data_unbalanced = pd.read_csv('dataset/before_softbal_label_twit.csv')
    twitter_data_balanced = pd.read_csv('dataset/label_twit.csv')

    # Calculate sentiment counts for Instagram and Twitter
    instagram_sentiment_count_unbalanced = instagram_data_unbalanced['Sentiment'].value_counts()
    instagram_sentiment_count_balanced = instagram_data_balanced['Sentiment'].value_counts()
    twitter_sentiment_count_unbalanced = twitter_data_unbalanced['Sentiment'].value_counts()
    twitter_sentiment_count_balanced = twitter_data_balanced['Sentiment'].value_counts()

    # Generate WordClouds for Instagram (Balanced)
    instagram_wordcloud_b64_pos_balanced = generate_wordcloud(instagram_data_balanced, 'positif')
    instagram_wordcloud_b64_neg_balanced = generate_wordcloud(instagram_data_balanced, 'negatif')
    instagram_wordcloud_b64_neu_balanced = generate_wordcloud(instagram_data_balanced, 'netral')
    instagram_wordcloud_b64_total_balanced = generate_total_wordcloud(instagram_data_balanced)

    # Generate WordClouds for Instagram (Unbalanced)
    instagram_wordcloud_b64_pos_unbalanced = generate_wordcloud(instagram_data_unbalanced, 'positif')
    instagram_wordcloud_b64_neg_unbalanced = generate_wordcloud(instagram_data_unbalanced, 'negatif')
    instagram_wordcloud_b64_neu_unbalanced = generate_wordcloud(instagram_data_unbalanced, 'netral')
    instagram_wordcloud_b64_total_unbalanced = generate_total_wordcloud(instagram_data_unbalanced)

    # Generate WordClouds for Twitter (Balanced)
    twitter_wordcloud_b64_pos_balanced = generate_wordcloud(twitter_data_balanced, 'positif')
    twitter_wordcloud_b64_neg_balanced = generate_wordcloud(twitter_data_balanced, 'negatif')
    twitter_wordcloud_b64_neu_balanced = generate_wordcloud(twitter_data_balanced, 'netral')
    twitter_wordcloud_b64_total_balanced = generate_total_wordcloud(twitter_data_balanced)

    # Generate WordClouds for Twitter (Unbalanced)
    twitter_wordcloud_b64_pos_unbalanced = generate_wordcloud(twitter_data_unbalanced, 'positif')
    twitter_wordcloud_b64_neg_unbalanced = generate_wordcloud(twitter_data_unbalanced, 'negatif')
    twitter_wordcloud_b64_neu_unbalanced = generate_wordcloud(twitter_data_unbalanced, 'netral')
    twitter_wordcloud_b64_total_unbalanced = generate_total_wordcloud(twitter_data_unbalanced)

    # Generate Sentiment Distribution Diagrams (Instagram)
    instagram_img_b64_unbalanced = generate_sentiment_distribution(instagram_data_unbalanced)
    instagram_img_b64_balanced = generate_sentiment_distribution(instagram_data_balanced)

    # Generate Sentiment Distribution Diagrams (Twitter)
    twitter_img_b64_unbalanced = generate_sentiment_distribution(twitter_data_unbalanced)
    twitter_img_b64_balanced = generate_sentiment_distribution(twitter_data_balanced)

    # Render the HTML template and pass all the necessary data
    return render_template(
        'visualization.html',
        instagram_img_b64_unbalanced=instagram_img_b64_unbalanced,
        instagram_img_b64_balanced=instagram_img_b64_balanced,
        twitter_img_b64_unbalanced=twitter_img_b64_unbalanced,
        twitter_img_b64_balanced=twitter_img_b64_balanced,
        instagram_wordcloud_b64_pos_balanced=instagram_wordcloud_b64_pos_balanced,
        instagram_wordcloud_b64_neg_balanced=instagram_wordcloud_b64_neg_balanced,
        instagram_wordcloud_b64_neu_balanced=instagram_wordcloud_b64_neu_balanced,
        instagram_wordcloud_b64_total_balanced=instagram_wordcloud_b64_total_balanced,
        instagram_wordcloud_b64_pos_unbalanced=instagram_wordcloud_b64_pos_unbalanced,
        instagram_wordcloud_b64_neg_unbalanced=instagram_wordcloud_b64_neg_unbalanced,
        instagram_wordcloud_b64_neu_unbalanced=instagram_wordcloud_b64_neu_unbalanced,
        instagram_wordcloud_b64_total_unbalanced=instagram_wordcloud_b64_total_unbalanced,
        twitter_wordcloud_b64_pos_balanced=twitter_wordcloud_b64_pos_balanced,
        twitter_wordcloud_b64_neg_balanced=twitter_wordcloud_b64_neg_balanced,
        twitter_wordcloud_b64_neu_balanced=twitter_wordcloud_b64_neu_balanced,
        twitter_wordcloud_b64_total_balanced=twitter_wordcloud_b64_total_balanced,
        twitter_wordcloud_b64_pos_unbalanced=twitter_wordcloud_b64_pos_unbalanced,
        twitter_wordcloud_b64_neg_unbalanced=twitter_wordcloud_b64_neg_unbalanced,
        twitter_wordcloud_b64_neu_unbalanced=twitter_wordcloud_b64_neu_unbalanced,
        twitter_wordcloud_b64_total_unbalanced=twitter_wordcloud_b64_total_unbalanced,
        instagram_sentiment_count_unbalanced=instagram_sentiment_count_unbalanced,
        instagram_sentiment_count_balanced=instagram_sentiment_count_balanced,
        twitter_sentiment_count_unbalanced=twitter_sentiment_count_unbalanced,
        twitter_sentiment_count_balanced=twitter_sentiment_count_balanced
    )







if __name__ == '__main__':
    app.run(debug=True)
