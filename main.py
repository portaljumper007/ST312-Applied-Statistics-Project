import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import stft
from scipy.ndimage import gaussian_filter
import os
import pickle
import charts_NLP_tristan_filter
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import statsmodels.api as sm

def load_or_create_topic_strengths(filename, num_topics):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        # Generate data
        weekly_topic_strengths_data = charts_NLP_tristan_filter.main(show_graph=True, num_topics=num_topics)
        # Extract and save only the necessary data from weekly_topic_strengths
        data_to_save = {week: {topic: strength for topic, strength in topics.items()} 
                        for week, topics in weekly_topic_strengths_data.items()}
        with open(filename, 'wb') as file:
            pickle.dump(data_to_save, file)
        return weekly_topic_strengths_data

def get_week_year(date):
    return f"{date.isocalendar()[0]}-W{date.isocalendar()[1]}"

def main():
    for location in ["Chicago","LA"]:
        num_topics = 10
        weekly_topic_strengths = load_or_create_topic_strengths('weekly_topic_strengths.pkl', num_topics)

        # Load and preprocess weather data
        weather_df = pd.read_csv(location+" Weather.csv")
        if location == "LA":
            weather_df['DATE'] = pd.to_datetime(weather_df['DATE'], format='%d/%m/%Y') #LA
        elif location == "Chicago":
            weather_df['DATE'] = pd.to_datetime(weather_df['DATE'], format='%Y-%m-%d') #CHICAGO
        weather_df.set_index('DATE', inplace=True)

        # Select only numeric columns relevant to the analysis
        numeric_columns = ['PRCP', 'SNOW', 'TMAX', 'TMIN']
        weather_numeric = weather_df[numeric_columns]
        weekly_weather = weather_numeric.groupby(pd.Grouper(freq='W')).mean()
        weekly_weather_for_autocorr = weekly_weather.copy()
        weekly_weather_for_autocorr.fillna(0, inplace=True)

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
            # Filling NaNs with zeros
            strengths_filled = np.nan_to_num(strengths)
            f, t, Zxx = stft(strengths_filled, window='hamming', nperseg=window_size)
            #print(f"STFT Output Shape for Topic {topic}: {Zxx.shape}, Time Length: {len(t)}")
            autocorr = [np.correlate(Zxx[:,i], weekly_weather_for_autocorr['TMAX'], mode='same') for i in range(len(t))]
            autocorr_sum = [np.sum(np.abs(a)) for a in autocorr]
            autocorrelation_results[topic] = autocorr_sum
            #print(f"Autocorrelation data for topic {topic}:", autocorr_sum)

        weekly_topic_strengths_aggregated = defaultdict(lambda: np.zeros(53))
        for date, strengths in weekly_topic_strengths.items():
            week_of_year = pd.Timestamp(date).isocalendar()[1]
            for topic in range(num_topics):
                weekly_topic_strengths_aggregated[topic][week_of_year - 1] += strengths.get(topic, 0)

        fig = make_subplots(rows=3, cols=4, specs=[[{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]],
                            subplot_titles=("Topic Strength Over Time", "Autocorrelation of Topic Strengths with Max Temperature", "Smoothed NLP Strengths by Week of the Year", "NLP Strengths by Week of the Year", "Precipitation vs Topic Strength", "Max Temperature vs Topic Strength", "Snow vs Topic Strength", "Min Temperature vs Topic Strength", "LOESS: Precipitation vs Topic Strength", "LOESS: Max Temperature vs Topic Strength", "LOESS: Snow vs Topic Strength", "LOESS: Min Temperature vs Topic Strength"))

        # Autocorrelation of Topic Strengths with Max Temperature on the first row
        for topic, autocorr in autocorrelation_results.items():
            fig.add_trace(go.Scattergl(x=weekly_weather.index, y=autocorr, mode='lines', name=f'Topic {topic + 1} Autocorrelation'), row=1, col=2)

        # Initialize a dictionary to hold time series data for each topic
        topic_time_series = {topic: [] for topic in range(num_topics)}
        dates = sorted(weekly_topic_strengths.keys())
        # Aggregate data for each topic
        for date in dates:
            for topic in range(num_topics):
                strength = weekly_topic_strengths[date].get(topic, 0)
                topic_time_series[topic].append((date, strength))
        # Plot each topic's time series data
        for topic, data in topic_time_series.items():
            dates, strengths = zip(*data)  # Unpack the tuples into two lists
            fig.add_trace(go.Scattergl(x=dates, y=strengths, mode='lines', name=f'Topic {topic} Strength Over Time'), row=1, col=1)

        # Ensure "NLP Strengths by Week of the Year" plots aggregated data correctly
        for topic, strengths in weekly_topic_strengths_aggregated.items():
            fig.add_trace(go.Scattergl(x=np.arange(1, 54), y=strengths, mode='lines', name=f'Topic {topic + 1} NLP Strengths by Week'), row=1, col=4)

        # Smoothed NLP Strengths by Week of the Year, third graph on the first row
        for topic, strengths in weekly_topic_strengths_aggregated.items():
            smoothed_strengths = gaussian_filter(strengths, sigma=2)
            fig.add_trace(go.Scattergl(x=np.arange(1, 54), y=smoothed_strengths, mode='lines', name=f'Topic {topic + 1} Smoothed'), row=1, col=3)

        # Scatter graphs and fitted lines, maintaining the color consistency
        topic_colors = px.colors.qualitative.Plotly
        for i, feature in enumerate(['PRCP', 'TMAX', 'SNOW', 'TMIN']):
            for topic in range(num_topics):
                color = topic_colors[topic % len(px.colors.qualitative.Plotly)]
                x = weekly_weather[feature].values
                y = np.array(topic_strengths_aligned[topic])
                fig.add_trace(go.Scattergl(x=x, y=y, mode='markers', marker=dict(color=color, opacity=0.25), name=f'Topic {topic+1} vs {feature}'), row=2, col=i+1)
                # Filter out NaN values which can cause errors in LOESS
                valid_indices = ~np.isnan(x) & ~np.isnan(y)
                x_filtered, y_filtered = x[valid_indices], y[valid_indices]
                # Apply LOESS smoothing
                lowess = sm.nonparametric.lowess(y_filtered, x_filtered, frac=0.25)
                # lowess result is a two-column array, first column is x, second column is y
                x_lowess, y_lowess = lowess[:, 0], lowess[:, 1]
                fig.add_trace(go.Scattergl(x=x_lowess, y=y_lowess, mode='lines', line=dict(color=color), name=f'LOESS Fit Topic {topic+1}'), row=3, col=i+1)

        # Update axes and layout
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Week of the Year", row=1, col=3)
        fig.update_xaxes(title_text="Precipitation (in)", row=2, col=1)
        fig.update_xaxes(title_text="Max Temperature (°F)", row=2, col=2)
        fig.update_xaxes(title_text="Snow (in)", row=2, col=3)
        fig.update_xaxes(title_text="Min Temperature (°F)", row=2, col=4)
        fig.update_layout(height=1800, title_text=f"Exploratory Analysis of Weather Trends and NLP of Music Charts - {location}")
        fig.show()


if __name__ == '__main__':
    main()