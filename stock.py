import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from newsapi.newsapi_client import NewsApiClient
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from datetime import date, timedelta
from pandas_datareader import data as pdr
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence

classifier = TextClassifier.load('en-sentiment')

def get_sentiment_flair(text):
    # Make a sentence object
    sentence = Sentence(text)

    # Predict the sentiment
    classifier.predict(sentence)

    # Extract the sentiment
    sentiment = sentence.labels[0].value  # 'POSITIVE' or 'NEGATIVE'
    score = sentence.labels[0].score  # confidence score

    return sentiment, score

# Initialize NewsApiClient
newsapi = NewsApiClient(api_key='f2fb7f58a9904a528f5d5d9c66c5682e')

# Function to fetch news data
from datetime import datetime, timedelta

def fetch_news(company):
    # Calculate dates for past month
    end_date = datetime.now()
    start_date = end_date - timedelta(30)

    class sources_news_fetch:    
        def __init__(self):
            print("init")

        def getSources(self):
            newsapi.get_sources(category='business',language='en')

    sources_news = sources_news_fetch().getSources()


    # All articles upto 100 articles
    all_articles = newsapi.get_everything(q=company,
                                          sources=sources_news,
                                          qintitle=company,
                                          from_param=start_date.strftime('%Y-%m-%d'),
                                          to=end_date.strftime('%Y-%m-%d'),
                                          language='en',
                                          sort_by='publishedAt')
    
    all_articles = all_articles['articles']

    # Convert articles to DataFrame
    df_articles = pd.DataFrame(all_articles)
    
    # Drop columns if they exist in DataFrame
    columns_to_drop = ['source', 'url', 'urlToImage', 'description','content']
    df_articles = df_articles.drop(columns=[col for col in columns_to_drop if col in df_articles.columns], errors='ignore')

    # Extract date from publishedAt
    if 'publishedAt' in df_articles.columns:
        df_articles['publishedAt'] = pd.to_datetime(df_articles['publishedAt'])
        df_articles['date'] = df_articles['publishedAt'].dt.date
        df_articles = df_articles.drop(columns=['publishedAt'], errors='ignore')

    # Perform sentiment analysis
    sia = SentimentIntensityAnalyzer()
    df_articles['textBlob_sentiment'] = df_articles['title'].apply(lambda title: TextBlob(title).sentiment.polarity)
    df_articles['vader_sentiment'] = df_articles['title'].apply(lambda title: sia.polarity_scores(title)['compound'])
    df_articles['flair_sentiment'] = df_articles['title'].apply(lambda title: get_sentiment_flair(title)[0])
    df_articles['flair_score'] = df_articles['title'].apply(lambda title: get_sentiment_flair(title)[1])

    return df_articles

# Function to fetch stock data and predict
def fetch_stock_data(ticker):
    start = '2010-01-01'
    today = date.today()
    yesterday = today - timedelta(days=1)
    end = yesterday

    yf.pdr_override()
    df = pdr.get_data_yahoo(ticker, start=start, end=end)
    df = df.reset_index()
    df = df.drop(['Date','Adj Close'], axis=1)

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.60)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.60):int(len(df))])

    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = scaler.fit_transform(data_training)

    x_train = []
    y_train = []

    for i in range(100, data_training.shape[0]):
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i,0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Load model
    model = load_model('keras_model.h5')

    # Machine Learning Model

    # model = Sequential()
    # model.add(LSTM(units = 90, activation = 'softsign', return_sequences = True, input_shape = (x_train.shape[1],1)))
    # model.add(Dropout(0.2))

    # model.add(LSTM(units = 100, activation = 'softsign', return_sequences = True))
    # model.add(Dropout(0.3))

    # model.add(LSTM(units = 120, activation = 'softsign', return_sequences = True))
    # model.add(Dropout(0.4))

    # model.add(LSTM(units = 160, activation = 'softsign'))
    # model.add(Dropout(0.5))

    # model.add(Dense(units = 1))

    # model.compile(optimizer='RMSprop', loss = 'mean_squared_error')
    # model.fit(x_train, y_train, epochs=10)
    # model.save('keras_model.h5')

    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Making Predictions

    y_predicted = model.predict(x_test)
    scaler = scaler.scale_

    scale_factor = 1/scaler[0]
    y_predicted = y_predicted*scale_factor
    y_test = y_test*scale_factor

    mape = mean_absolute_error(y_test, y_predicted)

    # Plotting Closing Price v/s Time Chart with 50MA and 200MA
    st.subheader('Closing Price v/s Time Chart with 50MA and 200MA')
    fig2 = plt.figure(figsize = (12,6))
    ma50 = df.Close.rolling(50).mean()
    ma200 = df.Close.rolling(200).mean()
    plt.plot(df.Close)
    plt.plot(ma50, 'r')
    plt.plot(ma200, 'g')
    st.pyplot(fig2)

    return y_test, y_predicted,mape

def calculate_correlation(ticker, news_data):
    # Converting news_data columns to datetime
    news_data['date'] = pd.to_datetime(news_data['date'])

    # Fetch stock data for the last month for correlation calculation
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    df_last_month = yf.download(ticker, start=start_date, end=end_date)
    df_last_month.reset_index(inplace=True)
    df_last_month.rename(columns={'Date':'date', 'Close':'Actual Price'}, inplace=True)
    df_last_month['date'] = pd.to_datetime(df_last_month['date'])  # Ensure 'date' is in datetime format

    # Merge last month's stock prices and news data
    merged_data = pd.merge(df_last_month, news_data, on='date', how='inner')

    # Calculate and display correlation between last month's stock prices and sentiment scores

    # TextBlob Correlation
    correlation_textBlob = merged_data['Actual Price'].corr(merged_data['textBlob_sentiment'])

    # NLTK's Vader Correlation
    correlation_vader = merged_data['Actual Price'].corr(merged_data['vader_sentiment'])

    # Flair Correlation
    correlation_flair = merged_data['Actual Price'].corr(merged_data['flair_score'])

    # Average correlation
    weight_flair = 0.34
    weight_vader = 0.33
    weight_textblob = 0.33
    merged_data['weighted_average_score'] = weight_flair * merged_data['flair_score'] + weight_vader * merged_data['vader_sentiment'] + weight_textblob * merged_data['textBlob_sentiment']
    average_score = merged_data['Actual Price'].corr(merged_data['weighted_average_score'])

    # Plotting the scores
    # Convert date from object to datetime
    merged_data['date'] = pd.to_datetime(merged_data['date'])

    # Plot sentiment scores over time
    fig, ax1 = plt.subplots(figsize=(12,6))

    color = 'tab:blue'
    # Sentiment scores
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sentiment Score', color=color)
    ax1.plot(merged_data['date'], merged_data['weighted_average_score'], color=color, label='Average Sentiment score')
    ax1.plot(merged_data['date'], merged_data['textBlob_sentiment'], color='tab:cyan', label='TextBlob Sentiment score')
    ax1.plot(merged_data['date'], merged_data['vader_sentiment'], color='tab:orange', label='VADER Sentiment score')
    ax1.plot(merged_data['date'], merged_data['flair_score'], color='tab:green', label='Flair Sentiment score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx() 

    color = 'tab:red'
    # Stock prices
    ax2.set_ylabel('Stock Price', color=color)
    ax2.plot(merged_data['date'], merged_data['Actual Price'], color=color, label='Actual Stock Price')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    st.pyplot(fig)


    return correlation_textBlob, correlation_vader, correlation_flair,average_score, merged_data


# The companies you can find data of
company_ticker_mapping = {
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'Google': 'GOOGL',
}

# User inputs
# company = st.text_input('Enter company name', 'Apple')
# ticker = st.text_input('Enter Stock Ticker', 'AAPL')
company = st.selectbox('Choose company name', list(company_ticker_mapping.keys()))
ticker = company_ticker_mapping[company]
st.text_input(label="Ticker",placeholder=ticker,disabled=True)

# Fetch data
news_data = fetch_news(company)
actual_prices, predicted_prices,mape = fetch_stock_data(ticker)

# Display actual and predicted stock prices
st.subheader('Actual and Predicted Stock Prices')
fig = plt.figure(figsize=(12,6))
plt.plot(actual_prices, color='blue', label='Actual Stock Price')
plt.plot(predicted_prices, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig)

# Error
st.subheader("Error margin in the prediction")
st.text_input(label="Mean absolute error",placeholder=mape,disabled=True)

# Display news data and sentiment scores
st.subheader('News Data and Sentiment Scores')
st.write(news_data)

#  Calculate and display correlation between last month's stock prices and sentiment scores
st.subheader('Correlation Between Stock Prices and News Sentiment')
correlation_textBlob, correlation_vader, correlation_flair,average_score, merged_data = calculate_correlation(ticker, news_data)

# TextBlob Correlation
if np.isnan(correlation_textBlob):
    st.text_input(label="TextBlob Correlation score",placeholder="No textBlob correlation found",disabled=True)
else:
    st.text_input(label="TextBlob Correlation score",placeholder=correlation_textBlob,disabled=True)  

# NLTK's Vader Correlation  
if np.isnan(correlation_vader):
    st.text_input(label="Vader Correlation score",placeholder="No vader correlation found",disabled=True)
else:
    st.text_input(label="Vader Correlation score",placeholder=correlation_vader,disabled=True)

# Flair Correlation  
if np.isnan(correlation_flair):
    st.text_input(label="Flair Correlation score",placeholder="No flair correlation found",disabled=True)
else:
    st.text_input(label="Flair Correlation score",placeholder=correlation_flair,disabled=True)

# Average Correlation
if np.isnan(average_score):
    st.text_input(label="Average Correlation score",placeholder="No average correlation found",disabled=True)
else:
    st.text_input(label="Average Correlation score",placeholder=average_score,disabled=True)

# Range and variance of sentiment scores
st.subheader('Sentiment range and variance')

# TextBlob
sentiment_range_textBlob = news_data['textBlob_sentiment'].max() - news_data['textBlob_sentiment'].min()
sentiment_variance_textBlob = news_data['textBlob_sentiment'].var()
st.text_input(label="TextBlob Sentiment score range",placeholder=sentiment_range_textBlob,disabled=True)
st.text_input(label="TextBlob score Variance",placeholder=sentiment_variance_textBlob,disabled=True)

# NLTK's Vader
sentiment_range_vader = news_data['vader_sentiment'].max() - news_data['vader_sentiment'].min()
sentiment_variance_vader = news_data['vader_sentiment'].var()
st.text_input(label="Vader Sentiment score range",placeholder=sentiment_range_vader,disabled=True)
st.text_input(label="Vader score Variance",placeholder=sentiment_variance_vader,disabled=True)

# Flair
sentiment_range_flair = news_data['flair_score'].max() - news_data['flair_score'].min()
sentiment_variance_flair = news_data['flair_score'].var()
st.text_input(label="Flair Sentiment score range",placeholder=sentiment_range_flair,disabled=True)
st.text_input(label="Flair score Variance",placeholder=sentiment_variance_flair,disabled=True)


# Checks

# Check if merged_data is empty
if merged_data.empty:
    st.write('No overlap in dates between stock prices and news data.')

# Check for nan values in merged_data
if merged_data['Actual Price'].isna().any() or merged_data['textBlob_sentiment'].isna().any() or merged_data['vader_sentiment'].isna().any() or merged_data['flair_score'].isna().any():
    st.write('Nan values in Actual Price or sentiment.')