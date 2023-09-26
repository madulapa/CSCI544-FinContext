import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import talib
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import nltk

nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def convert_tag(tag):
    if tag[0] == 'J':
        return wordnet.ADJ
    elif tag[0] == 'V':
        return wordnet.VERB
    elif tag[0] == 'N':
        return wordnet.NOUN
    elif tag[0] == 'R':
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize(text, lemmatizer):
    words_tags = pos_tag(word_tokenize(text))
    l_words = [lemmatizer.lemmatize(word_tag[0], convert_tag(
        word_tag[1])) for word_tag in words_tags]
    return " ".join(l_words)


class MultimodalModel(nn.Module):
    def __init__(self, hidden_size, num_classes, extraction_type, lstm_or_gru='lstm', num_data_size=10):
        super(MultimodalModel, self).__init__()
        self.num_data_size = num_data_size
        self.extraction_type = extraction_type

        if self.extraction_type == 'bert':
            self.text_model = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif self.extraction_type == 'finbert':
            self.text_model = BertModel.from_pretrained('ProsusAI/finbert')
            self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')

        self.text_model.eval()

        if lstm_or_gru == 'lstm':
            self.recurrent_model = nn.LSTM(
                input_size=self.num_data_size, hidden_size=hidden_size, batch_first=True)
        else:
            self.recurrent_model = nn.GRU(
                input_size=self.num_data_size, hidden_size=hidden_size, batch_first=True)

        self.fc1 = nn.Linear(
            hidden_size + self.text_model.config.hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, article_input, numerical_data):
        if self.extraction_type in ['bert', 'finbert']:
            inputs = self.tokenizer(
                article_input, padding='max_length', truncation=True, return_tensors="pt"
            )
            text_outputs = self.text_model(**inputs)
            text_embed = text_outputs.last_hidden_state[:, 0, :]
        else:
            text_embed = article_input

        # if isinstance(self.recurrent_model, nn.LSTM):
        #     recurrent_output, _ = self.recurrent_model(numerical_data.unsqueeze(1))
        # else:
        # print(numerical_data, numerical_data.unsqueeze(1))
        recurrent_output, _ = self.recurrent_model(
            numerical_data.unsqueeze(1))

        recurrent_output = recurrent_output[:, -1, :]

        combined = torch.cat([text_embed, recurrent_output], dim=1)

        output = self.fc1(combined)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.softmax(output)

        return output


class StockDataset(Dataset):
    def __init__(self, df, extraction_type, lookback=14):
        self.data = df
        self.lookback = lookback
        self.extraction_type = extraction_type
        self.preprocess()
        if self.extraction_type == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=3000)
            self.tfidf = vectorizer.fit_transform(
                list(self.data['Article Text']))
        elif self.extraction_type == 'w2v':
            self.word2vec = api.load('word2vec-google-news-300')

        self.calculate_technical_indicators()

    def preprocess(self):
        self.data['Article Text'] = self.data['Article Text'].apply(
            lambda x: x.lower())
        myLemmatizer = WordNetLemmatizer()
        self.data['Article Text'] = self.data['Article Text'].apply(
            lambda x: lemmatize(x, myLemmatizer))

    def calculate_technical_indicators(self):
        self.data['SMA'] = talib.SMA(
            self.data['Stock Close'], timeperiod=self.lookback)
        self.data['EMA'] = talib.EMA(
            self.data['Stock Close'], timeperiod=self.lookback)
        self.data['RSI'] = talib.RSI(
            self.data['Stock Close'], timeperiod=self.lookback)
        self.data['MACD'], _, _ = talib.MACD(
            self.data['Stock Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        self.data['ATR'] = talib.ATR(
            self.data['Stock High'], self.data['Stock Low'], self.data['Stock Close'], timeperiod=self.lookback)
        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]

        article_text = row['Article Text']
        if self.extraction_type == 'tfidf':
            article_input = torch.tensor(
                self.tfidf[idx].toarray(), dtype=torch.float32)
        elif self.extraction_type == 'w2v':
            article_input = []
            for word in word_tokenize(article_text):
                if word in self.word2vec.vocab:
                    article_input.append(self.word2vec[word])
                else:
                    article_input.append([0 for _ in range(300)])
            article_input = torch.tensor(np.array(article_input)[:512])
            article_input = article_input.type(torch.FloatTensor)
        elif self.extraction_type in ['bert', 'finbert']:
            article_input = article_text

        stock_numerical_data = row[[
            'Stock Open',
            'Stock Close',
            'Stock High',
            'Stock Low',
            'volume',
            'SMA',
            'EMA',
            'MACD',
            'RSI',
            'ATR'
        ]].values
        stock_numerical_data = torch.tensor(
            stock_numerical_data.tolist(), dtype=torch.float32)
        sentiment_mapping = {
            "Bearish": 0,
            "Somewhat-Bearish": 1,
            "Neutral": 2,
            "Somewhat-Bullish": 3,
            "Bullish": 4,
        }
        if 'overall_sentiment_label' in self.data.columns:
            label = torch.tensor(
                sentiment_mapping[row['overall_sentiment_label']], dtype=torch.long)
        else:
            label = None

        return article_input, stock_numerical_data, label
