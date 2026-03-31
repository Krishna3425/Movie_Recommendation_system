from textblob import TextBlob

class SentimentAnalyzer:
    def analyze_sentiment(self, text):
        if not text:
            return "Neutral", 0.0
            
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return sentiment, round(polarity, 2)

    def get_batch_sentiment(self, text_list):
        results = []
        for text in text_list:
            results.append(self.analyze_sentiment(text))
        return results
