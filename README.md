# Fake News Detection Pipeline

An automated pipeline for detecting fake news using BERT and Airflow. This project combines deep learning with automated workflow management to continuously monitor and classify news articles.

## Features

- BERT-based fake news classification
- Automated news fetching from multiple sources
- Real-time processing pipeline using Apache Airflow
- MongoDB integration for data storage
- Automated alerts via email and Slack
- Comprehensive evaluation metrics

## Architecture

```
├── fake_news_dag.py    # Airflow DAG definition
├── utils.py            # Utility functions
├── config.py          # Configuration settings
└── requirements.txt    # Project dependencies
```

## Prerequisites

- Python 3.8+
- Apache Airflow
- MongoDB
- PyTorch
- Transformers (Hugging Face)
- News API key
- SMTP server access (for email alerts)
- Slack webhook (for Slack alerts)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nishusinha20/fake-news-detector.git
cd fake-news-detector
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure settings:
   - Update `config.py` with your credentials
   - Set up MongoDB connection
   - Configure News API key
   - Set up email and Slack notifications

5. Initialize Airflow:
```bash
export AIRFLOW_HOME=~/airflow
airflow db init
```

6. Start Airflow services:
```bash
airflow webserver --port 8080
airflow scheduler
```

## Usage

1. Access Airflow UI at `http://localhost:8080`
2. Enable the `fake_news_detection` DAG
3. Monitor the pipeline execution
4. Check MongoDB for results
5. Configure alerts as needed

## Pipeline Steps

1. `fetch_news_data`: Fetches news from configured sources
2. `save_raw_to_storage`: Stores raw data in MongoDB
3. `preprocess_news_data`: Cleans and preprocesses text
4. `classify_with_bert`: Performs BERT-based classification
5. `save_results_db`: Stores results in MongoDB
6. `send_alerts_if_fake`: Sends alerts for detected fake news

## Model Training

The BERT model should be trained separately using the provided notebook. The trained model should be saved as `fake_news_model.pth` in the project directory.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuggingFace Transformers library
- Apache Airflow
- News API
- MongoDB 