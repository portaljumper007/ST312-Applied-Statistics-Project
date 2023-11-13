import pandas as pd
import matplotlib.pyplot as plt

# Load "Music Charts.csv"
music_df = pd.read_csv("Music Charts.csv")  # Default delimiter is comma
print(music_df.columns)
# Convert WeekID to datetime format
music_df['WeekID'] = pd.to_datetime(music_df['WeekID'])

# Load "LA weather.csv"
weather_df = pd.read_csv("LA weather.csv")

print(weather_df.columns)
# Convert DATE to datetime format
weather_df['DATE'] = pd.to_datetime(weather_df['DATE'], format='%d/%m/%Y')

# Plotting the peak position of a particular song over time
song_name = "Don't Just Stand There"
filtered_music_df = music_df[music_df['Song'] == song_name]
plt.figure(figsize=(10, 6))
plt.plot(filtered_music_df['WeekID'], filtered_music_df['Peak Position'])
plt.title(f"Peak Position of {song_name} Over Time")
plt.xlabel("WeekID")
plt.ylabel("Peak Position")
plt.show()  # Plots peak positions of the selected song

# Plotting the average temperature in LA for a given range
filtered_weather_df = weather_df[(weather_df['DATE'] >= '2017-09-26') & (weather_df['DATE'] <= '2017-09-30')]
plt.figure(figsize=(10, 6))
plt.plot(filtered_weather_df['DATE'], filtered_weather_df['TAVG'])
plt.title("Average Temperature in LA (2017-09-26 to 2017-09-30)")
plt.xlabel("Date")
plt.ylabel("Average Temperature")
plt.show()  # Plots average temperature in LA for the selected date range

# Plotting the number of weeks a song has been on the chart for top 10 songs
top_10_songs = music_df.groupby('Song').size().nlargest(10).index.tolist()
top_10_df = music_df[music_df['Song'].isin(top_10_songs)]
weeks_on_chart = top_10_df.groupby('Song')['Weeks on Chart'].max()
plt.figure(figsize=(10, 6))
weeks_on_chart.plot(kind='bar')
plt.title("Number of Weeks Top 10 Songs Have Been on Chart")
plt.xlabel("Song")
plt.ylabel("Weeks on Chart")
plt.show()  # Plots number of weeks for top 10 songs on the chart

