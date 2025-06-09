from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
import requests
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import json
from pymongo import MongoClient
import os
from bs4 import BeautifulSoup

# Initialize BERT model and tokenizer (same as in notebook)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MongoDB connection
MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
DB_NAME = "fake_news_db"

def fetch_news_data(**context):
    """
    Fetch news articles from various sources
    """
    # Example: Fetching news from a sample news API
    news_sources = [
        "http://newsapi.org/v2/top-headlines?country=us&apiKey=YOUR_API_KEY",
        # Add more sources as needed
    ]
    
    articles = []
    for source in news_sources:
        try:
            response = requests.get(source)
            data = response.json()
            articles.extend(data.get('articles', []))
        except Exception as e:
            print(f"Error fetching from {source}: {str(e)}")
    
    # Save the context for next task
    context['task_instance'].xcom_push(key='raw_articles', value=articles)
    return len(articles)

def save_raw_to_storage(**context):
    """
    Save raw news data to storage
    """
    articles = context['task_instance'].xcom_pull(key='raw_articles')
    
    # Connect to MongoDB
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    raw_collection = db['raw_news']
    
    # Save raw articles with timestamp
    for article in articles:
        article['timestamp'] = datetime.now()
        raw_collection.insert_one(article)
    
    client.close()
    return len(articles)

def preprocess_news_data(**context):
    """
    Preprocess the news data
    """
    # Connect to MongoDB
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    raw_collection = db['raw_news']
    
    # Get recent unprocessed articles
    articles = list(raw_collection.find({'processed': {'$exists': False}}))
    
    processed_articles = []
    for article in articles:
        # Clean and preprocess text
        text = article.get('title', '') + ' ' + article.get('description', '')
        text = clean_text(text)
        
        # Tokenize using BERT tokenizer
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        processed_articles.append({
            'original_id': article['_id'],
            'text': text,
            'input_ids': encoding['input_ids'].tolist(),
            'attention_mask': encoding['attention_mask'].tolist()
        })
    
    context['task_instance'].xcom_push(key='processed_articles', value=processed_articles)
    client.close()
    return len(processed_articles)

def classify_with_bert(**context):
    """
    Classify news using BERT model
    """
    processed_articles = context['task_instance'].xcom_pull(key='processed_articles')
    
    # Load the trained model
    model = torch.load('fake_news_model.pth')
    model.eval()
    
    results = []
    with torch.no_grad():
        for article in processed_articles:
            input_ids = torch.tensor(article['input_ids']).to(device)
            attention_mask = torch.tensor(article['attention_mask']).to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
            
            results.append({
                'original_id': article['original_id'],
                'prediction': prediction,  # 0 for fake, 1 for real
                'confidence': confidence,
                'text': article['text']
            })
    
    context['task_instance'].xcom_push(key='classification_results', value=results)
    return len(results)

def save_results_db(**context):
    """
    Save classification results to database
    """
    results = context['task_instance'].xcom_pull(key='classification_results')
    
    # Connect to MongoDB
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    results_collection = db['classification_results']
    
    # Save results
    for result in results:
        result['timestamp'] = datetime.now()
        results_collection.insert_one(result)
    
    client.close()
    return len(results)

def send_alerts_if_fake(**context):
    """
    Send alerts for fake news articles
    """
    results = context['task_instance'].xcom_pull(key='classification_results')
    
    fake_news_count = 0
    for result in results:
        if result['prediction'] == 0 and result['confidence'] > 0.8:  # Fake news with high confidence
            # Here you could implement various alert mechanisms:
            # 1. Send email
            # 2. Push notification
            # 3. Slack message
            # 4. SMS alert
            print(f"ALERT: Potential fake news detected with {result['confidence']*100:.2f}% confidence")
            print(f"Text: {result['text'][:200]}...")
            fake_news_count += 1
    
    return fake_news_count

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['your-email@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'fake_news_detection',
    default_args=default_args,
    description='A DAG for fake news detection pipeline',
    schedule_interval=timedelta(hours=1),
    catchup=False
)

# Define tasks
start = DummyOperator(task_id='start', dag=dag)

fetch_news = PythonOperator(
    task_id='fetch_news_data',
    python_callable=fetch_news_data,
    provide_context=True,
    dag=dag,
)

save_raw = PythonOperator(
    task_id='save_raw_to_storage',
    python_callable=save_raw_to_storage,
    provide_context=True,
    dag=dag,
)

preprocess = PythonOperator(
    task_id='preprocess_news_data',
    python_callable=preprocess_news_data,
    provide_context=True,
    dag=dag,
)

classify = PythonOperator(
    task_id='classify_with_bert',
    python_callable=classify_with_bert,
    provide_context=True,
    dag=dag,
)

save_results = PythonOperator(
    task_id='save_results_db',
    python_callable=save_results_db,
    provide_context=True,
    dag=dag,
)

send_alerts = PythonOperator(
    task_id='send_alerts_if_fake',
    python_callable=send_alerts_if_fake,
    provide_context=True,
    dag=dag,
)

end = DummyOperator(task_id='end', dag=dag)

# Define task dependencies
start >> fetch_news >> save_raw >> preprocess >> classify >> save_results >> send_alerts >> end 