import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import stft
from scipy.ndimage import gaussian_filter
import os
import pickle
import charts_NLP
import matplotlib.dates as mdates

def load_or_create_topic_strengths(filename, num_topics):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        # Generate data
        weekly_topic_strengths_data = charts_NLP.main(show_graph=True, num_topics=num_topics)
        # Extract and save only the necessary data from weekly_topic_strengths
        data_to_save = {week: {topic: strength for topic, strength in topics.items()} 
                        for week, topics in weekly_topic_strengths_data.items()}
        with open(filename, 'wb') as file:
            pickle.dump(data_to_save, file)
        return weekly_topic_strengths_data

def get_week_year(date):
    return f"{date.isocalendar()[0]}-W{date.isocalendar()[1]}"

def main():
    num_topics = 5
    weekly_topic_strengths = load_or_create_topic_strengths('weekly_topic_strengths.pkl', num_topics)

    # Load and preprocess weather data
    weather_df = pd.read_csv("LA weather.csv")
    weather_df['DATE'] = pd.to_datetime(weather_df['DATE'], format='%d/%m/%Y')
    weather_df.set_index('DATE', inplace=True)

    # Select only numeric columns relevant to the analysis
    numeric_columns = ['PRCP', 'SNOW', 'TMAX', 'TMIN']
    weather_numeric = weather_df[numeric_columns]
    weekly_weather = weather_numeric.groupby(pd.Grouper(freq='W')).mean()

    # Normalize weekly_topic_strengths keys to week-year format
    normalized_topic_strengths = {get_week_year(pd.Timestamp(week)): strengths
                                  for week, strengths in weekly_topic_strengths.items()}

    # Aligning topic strengths with weather data
    topic_strengths_aligned = defaultdict(list)
    for week in weekly_weather.index:
        week_year = get_week_year(week)
        for topic in range(num_topics):
            if week_year in normalized_topic_strengths:
                strength = normalized_topic_strengths[week_year].get(topic, 0)
            else:
                strength = 0
            topic_strengths_aligned[topic].append(strength)

    # STFT analysis and autocorrelation
    window_size = 52*10  # Approx a year
    autocorrelation_results = defaultdict(list)

    for topic, strengths in topic_strengths_aligned.items():
        # Fill NaNs with zeros if needed
        strengths_filled = np.nan_to_num(strengths)

        f, t, Zxx = stft(strengths_filled, window='hamming', nperseg=window_size)
        print(f"STFT Output Shape for Topic {topic}: {Zxx.shape}, Time Length: {len(t)}")

        autocorr = [np.correlate(Zxx[:,i], weekly_weather['TMAX'], mode='same') for i in range(len(t))]
        autocorr_sum = [np.sum(np.abs(a)) for a in autocorr]
        autocorrelation_results[topic] = autocorr_sum

        print(f"Autocorrelation data for topic {topic}:", autocorr_sum)

    # Plotting the results with corrected time axis
    plt.figure(figsize=(15, 10))

    # Convert weekly_weather index to a list for mapping
    date_list = weekly_weather.index.to_list()

    # Number of STFT time frames per original data point
    frames_per_week = len(t) / len(date_list)

    for topic, autocorr in autocorrelation_results.items():
        if len(autocorr) == len(t):  # Ensure lengths match
            # Map STFT time frames to dates
            mapped_dates = [date_list[int(i // frames_per_week)] for i in range(len(t))]
            plt.plot(mapped_dates, autocorr, label=f'Topic {topic + 1}')

    plt.title("Autocorrelation of Topic Strengths with Max Temperature Over Time")
    plt.xlabel("Time")
    plt.ylabel("Autocorrelation Strength")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()  # Rotate date labels
    plt.legend()
    plt.show()

    ####aggregated day of year plot


    # New code for plotting NLP strengths against the day of the year
    daily_topic_strengths = defaultdict(lambda: np.zeros(366))  # Initialize array for each day of the year
    for date, strengths in weekly_topic_strengths.items():
        day_of_year = date.timetuple().tm_yday  # Get day of the year
        for topic in range(num_topics):
            daily_topic_strengths[topic][day_of_year - 1] += strengths.get(topic, 0)

    # Line graph for NLP strengths
    plt.figure(figsize=(15, 10))
    for topic, strengths in daily_topic_strengths.items():
        plt.plot(range(1, 367), strengths, label=f'Topic {topic + 1}')

    plt.title("NLP Strengths by Day of the Year")
    plt.xlabel("Day of the Year")
    plt.ylabel("Topic Strength")
    plt.legend()
    plt.show()

    # Smoothed line graph using Gaussian filter
    plt.figure(figsize=(15, 10))
    for topic, strengths in daily_topic_strengths.items():
        smoothed_strengths = gaussian_filter(strengths, sigma=5)  # Apply Gaussian filter for smoothing
        plt.plot(range(1, 367), smoothed_strengths, label=f'Topic {topic + 1}')

    plt.title("Smoothed NLP Strengths by Day of the Year (Gaussian Filter)")
    plt.xlabel("Day of the Year")
    plt.ylabel("Smoothed Topic Strength")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()