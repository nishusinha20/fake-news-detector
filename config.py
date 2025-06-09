# Database configuration
MONGO_CONFIG = {
    'CONNECTION_STRING': 'mongodb://localhost:27017/',
    'DB_NAME': 'fake_news_db',
    'COLLECTIONS': {
        'RAW_NEWS': 'raw_news',
        'PROCESSED_NEWS': 'processed_news',
        'RESULTS': 'classification_results'
    }
}

# News API configuration
NEWS_API_CONFIG = {
    'API_KEY': 'your-news-api-key',
    'ENDPOINTS': [
        'http://newsapi.org/v2/top-headlines?country=us',
        'http://newsapi.org/v2/everything?q=important'
    ]
}

# Model configuration
MODEL_CONFIG = {
    'MAX_LENGTH': 512,
    'BATCH_SIZE': 16,
    'MODEL_PATH': 'fake_news_model.pth',
    'CONFIDENCE_THRESHOLD': 0.8
}

# Alert configuration
ALERT_CONFIG = {
    'EMAIL': {
        'SMTP_SERVER': 'smtp.gmail.com',
        'SMTP_PORT': 587,
        'SENDER_EMAIL': 'your-email@gmail.com',
        'SENDER_PASSWORD': 'your-app-specific-password',
        'RECIPIENTS': ['recipient1@example.com', 'recipient2@example.com']
    },
    'SLACK': {
        'WEBHOOK_URL': 'your-slack-webhook-url',
        'CHANNEL': '#fake-news-alerts'
    }
}

# Airflow configuration
AIRFLOW_CONFIG = {
    'SCHEDULE_INTERVAL': '0 * * * *',  # Run every hour
    'RETRIES': 3,
    'RETRY_DELAY': 5,  # minutes
    'EMAIL_ON_FAILURE': True
} 