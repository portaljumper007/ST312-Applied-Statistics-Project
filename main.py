import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
from scipy.signal import stft
import charts_NLP

def main():
    # [Your existing NLP script here to create 'weekly_topic_strengths']
    num_topics = 5
    weekly_topic_strengths = charts_NLP.main(show_graph=False, num_topics=num_topics)

    # Load and preprocess weather data
    weather_df = pd.read_csv("LA weather.csv")
    weather_df['DATE'] = pd.to_datetime(weather_df['DATE'], format='%d/%m/%Y')

    # Set DATE as the index
    weather_df.set_index('DATE', inplace=True)

    # Select only numeric columns relevant to the analysis
    numeric_columns = ['PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN']  # Add or remove columns based on your dataset
    weather_numeric = weather_df[numeric_columns]

    # Group by week and calculate the mean of numeric columns
    weekly_weather = weather_numeric.groupby(pd.Grouper(freq='W')).mean()

    # Aligning topic strengths with weather data
    topic_strengths_aligned = defaultdict(list)
    for week in weekly_weather.index:
        for topic in range(num_topics):
            strength = weekly_topic_strengths.get(week, {}).get(topic, 0)
            topic_strengths_aligned[topic].append(strength)

    # STFT analysis and autocorrelation
    window_size = 52  # Approx a year
    autocorrelation_results = defaultdict(list)

    for topic, strengths in topic_strengths_aligned.items():
        f, t, Zxx = stft(strengths, window='hamming', nperseg=window_size)
        autocorr = [np.correlate(Zxx[:,i], weekly_weather['TAVG'], mode='same') for i in range(len(t))]
        autocorr_sum = [np.sum(np.abs(a)) for a in autocorr]
        autocorrelation_results[topic] = autocorr_sum

    # Plotting the results
    plt.figure(figsize=(15, 10))
    for topic, autocorr in autocorrelation_results.items():
        plt.plot(t, autocorr, label=f'Topic {topic + 1}')

    plt.title("Autocorrelation of Topic Strengths with Average Temperature Over Time")
    plt.xlabel("Time")
    plt.ylabel("Autocorrelation Strength")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()