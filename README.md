
# Stock and News Analysis 📈🗞️

Dive into the world of stocks with the **Stock and News Analysis** application. This Python-based tool not only allows users to fetch stock data but also analyzes recent news sentiment related to the stock. By combining stock price information with news sentiment, users can gain insights into the potential correlation between media coverage and stock performance.

## 🚀 Features

- **Stock Data Retrieval**: Fetch stock data for a specified company.
- **News Sentiment Analysis**: Utilize multiple sentiment analysis tools such as `TextBlob`, `NLTK's Vader`, `Flair`, and `Roberta` to gauge the sentiment of recent news articles.
- **Integrated News API**: Leverage the `newsapi` to fetch recent news articles related to the chosen stock.
- **Interactive UI**: Navigate and interact with the application using an intuitive interface powered by `streamlit`.
- **Data Visualization**: Witness the potential correlation between stock prices and news sentiment through visual graphs.

## 📁 File Structure

- 📄 `stock.py`: Core application logic encompassing stock data retrieval, sentiment analysis, and visualization.
- 📜 `keras_model.h5`: Pre-trained model, possibly related to sentiment analysis or stock prediction.

## 🔧 Setup & Execution

1. Install the necessary Python libraries:
   ```bash
   pip install yfinance pandas matplotlib streamlit torch plotly transformers newsapi keras nltk flair
   ```
2. Run the `streamlit` application:
   ```bash
   streamlit run stock.py
   ```
3. Access the application in your web browser and follow the on-screen instructions to choose a stock and view the analysis.

## 🧠 Technical Details

- **Sentiment Analysis Tools**: The application employs various tools and models like `TextBlob`, `NLTK's Vader`, `Flair`, and `Roberta` to achieve comprehensive sentiment analysis.
- **Stock Data**: Stock data retrieval is managed through the `yfinance` library.
- **Data Visualization**: The application uses libraries like `matplotlib` and `plotly` to visualize the correlation between stock prices and news sentiment.

## 🧪 Testing

1. Launch the application using the above execution steps.
2. Choose different stocks and observe the sentiment analysis results and stock data visualization.
3. Analyze the correlation graphs to understand the potential impact of news sentiment on stock prices.
