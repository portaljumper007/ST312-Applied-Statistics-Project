import pandas as pd
import matplotlib.pyplot as plt

def plot_weather_trends(weather_df):
    # Convert DATE to datetime format if not already done
    weather_df['DATE'] = pd.to_datetime(weather_df['DATE'], format='%d/%m/%Y')

    # Plotting Precipitation over time
    plt.figure(figsize=(10, 6))
    plt.plot(weather_df['DATE'], weather_df['PRCP'])
    plt.title("Precipitation Over Time")
    plt.xlabel("Date")
    plt.ylabel("Precipitation (inches)")
    plt.show()

    # Plotting Snow over time
    if 'SNOW' in weather_df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(weather_df['DATE'], weather_df['SNOW'])
        plt.title("Snowfall Over Time")
        plt.xlabel("Date")
        plt.ylabel("Snowfall (inches)")
        plt.show()

    # Plotting Average Temperature over time
    plt.figure(figsize=(10, 6))
    plt.plot(weather_df['DATE'], weather_df['TAVG'], label='Average Temperature')
    plt.plot(weather_df['DATE'], weather_df['TMAX'], label='Max Temperature')
    plt.plot(weather_df['DATE'], weather_df['TMIN'], label='Min Temperature')
    plt.title("Temperature Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel("Temperature (F)")
    plt.legend()
    plt.show()

def main():
    # Load "LA weather.csv"
    weather_df = pd.read_csv("LA weather.csv")

    # Call function to plot weather trends
    plot_weather_trends(weather_df)

if __name__ == '__main__':
    main()