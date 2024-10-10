import pandas as pd
import requests
import time
import schedule
import praw
from datetime import datetime, timedelta
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import logging

logging.basicConfig(level=logging.INFO)

# Initialize sentiment analysis model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

def fetch_social_media_data():
    crypto_subreddits = [
        "Bitcoin",
        "BTC",
        "Crypto",
        "CryptoNews"
    ]
    all_data = []
    for subreddit in crypto_subreddits:
        subreddit_data = extract_reddit_data(subreddit)
        if not subreddit_data.empty:
            all_data.append(subreddit_data)
    
    if not all_data:
        logging.error("No Reddit data collected")
        return None

    reddit_data = pd.concat(all_data, ignore_index=True)
    reddit_data = reddit_data[reddit_data['body'] != '']
    reddit_data = reddit_data.rename(columns={'body':'Description','comms_num':'No_Comments'})
    reddit_data['title'] = reddit_data['title'].apply(preprocess_text)
    reddit_data['Description'] = reddit_data['Description'].apply(preprocess_text)
    
    return reddit_data

def analyze_sentiment(text_data):
    try:
        inputs = tokenizer(text_data, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        sentiment_scores = 2 * probabilities[:, 1].numpy() - 1
        return sentiment_scores
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {str(e)}")
        return None

def extract_reddit_data(subreddit_name, limit=50):
    client_id = '09GDzbqdcnadRkM9t0_IWg'
    client_secret = 'z3H0rtRKu102jRhNL3U26iSsnW0MKQ'
    user_agent = 'Particular-Pay-2666'

    try:
        reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         user_agent=user_agent)
        subreddit = reddit.subreddit(subreddit_name)
        posts = []

        for post in subreddit.hot(limit=limit):
            posts.append({
                'title': post.title,
                'score': post.score,
                'id': post.id,
                'url': post.url,
                'comms_num': post.num_comments,
                'created': datetime.fromtimestamp(post.created),
                'body': post.selftext
            })

        return pd.DataFrame(posts)
    except Exception as e:
        logging.error(f"Error extracting data from r/{subreddit_name}: {str(e)}")
        return pd.DataFrame()

def preprocess_text(text):
    try:
        text = re.sub(r'[^a-zA-Z\s]', '', str(text))
        text = text.lower()
        return text
    except Exception as e:
        logging.error(f"Error processing text: {text[:50]}... Error: {str(e)}")
        return ""

def get_crypto_data(crypto_id, days=7):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data['prices'], columns=['timestamp', f'{crypto_id}_price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
        df.set_index('timestamp', inplace=True)
        
        df[f'{crypto_id}_volume'] = [x[1] for x in data['total_volumes']]
        df[f'{crypto_id}_market_cap'] = [x[1] for x in data['market_caps']]
        
        return df
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching crypto data for {crypto_id}: {str(e)}")
        return pd.DataFrame()

def calculate_volatility(df, crypto_id, window=7):
    df[f'{crypto_id}_returns'] = df[f'{crypto_id}_price'].pct_change()
    df[f'{crypto_id}_volatility'] = df[f'{crypto_id}_returns'].rolling(window=window).std() * (365**0.5)
    return df

def preprocess_data():
    reddit_data = fetch_social_media_data()
    if reddit_data is None:
        return None

    reddit_data['title_sentiment'] = analyze_sentiment(reddit_data['title'].tolist())
    reddit_data['description_sentiment'] = analyze_sentiment(reddit_data['Description'].tolist())

    reddit_data['created'] = pd.to_datetime(reddit_data['created'])
    reddit_daily = reddit_data.groupby(reddit_data['created'].dt.date).agg({
        'score': 'mean',
        'No_Comments': 'sum',
        'title_sentiment': 'mean',
        'description_sentiment': 'mean'
    }).reset_index()
    reddit_daily['created'] = pd.to_datetime(reddit_daily['created']).dt.date
    reddit_daily.set_index('created', inplace=True)

    btc_data = get_crypto_data('bitcoin')
    if btc_data.empty:
        logging.error("Failed to fetch Bitcoin data")
        return None

    btc_data = calculate_volatility(btc_data, 'bitcoin')
    btc_data['bitcoin_returns'] = btc_data['bitcoin_returns'].fillna(btc_data['bitcoin_returns'].mean())
    btc_data['bitcoin_volatility'] = btc_data['bitcoin_volatility'].fillna(btc_data['bitcoin_volatility'].mean())

    combined_data = pd.merge(btc_data, reddit_daily, left_index=True, right_index=True, how='left')
    logging.info('Data merger successful')

    combined_data['score'] = combined_data['score'].fillna(combined_data['score'].mean())
    combined_data['No_Comments'] = combined_data['No_Comments'].fillna(combined_data['No_Comments'].median())
    combined_data['title_sentiment'] = combined_data['title_sentiment'].fillna(combined_data['title_sentiment'].mean())
    combined_data['description_sentiment'] = combined_data['description_sentiment'].fillna(combined_data['description_sentiment'].mean())

    for col in ['score', 'No_Comments', 'title_sentiment', 'description_sentiment']:
        for i in range(1, 8):
            combined_data[f'{col}_lag{i}'] = combined_data[col].shift(i)

    for col in ['score_lag', 'No_Comments_lag', 'title_sentiment_lag', 'description_sentiment_lag']:
        for i in range(1, 8):
            combined_data[f"{col}{i}"] = combined_data[f"{col}{i}"].fillna(combined_data[f"{col}{i}"].mean())

    combined_data['sentiment'] = (combined_data['title_sentiment'] + combined_data['description_sentiment']) / 2
    for i in range(1, 8):
        combined_data[f"sentiment{i}"] = (combined_data[f'title_sentiment_lag{i}'] + combined_data[f'description_sentiment_lag{i}']) / 2

    sentiment_columns = ['sentiment'] + [f'sentiment{i}' for i in range(1, 8)]
    combined_data['total_sentiment'] = combined_data[sentiment_columns].mean(axis=1)

    combined_data.index.name = 'timestamp'
    combined_data = combined_data.reset_index()
    logging.info("Data processing completed successfully.")
    return combined_data

if __name__ == "__main__":
    processed_data = preprocess_data()
    if processed_data is not None:
        processed_data.to_csv('final_data.csv', index=False)
        print("Data saved to 'final_data.csv'")
    else:
        print("Data processing failed")