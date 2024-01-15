import pandas as pd
import matplotlib.pyplot as plt

music_charts = pd.read_csv("Music Charts.csv")
music_charts['WeekID'] = pd.to_datetime(music_charts['WeekID'])

# Histogram showing most popular song names
song_counts = music_charts['Song'].value_counts()
top_songs = song_counts.head(10)
plt.figure(figsize=(12, 6))
top_songs.plot(kind='bar', color='skyblue')
plt.title(f'Top 10 Most Popular Songs')
plt.xlabel('Song Names')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()

# Histogram showing most popular artists
song_counts = music_charts['Performer'].value_counts()
top_songs = song_counts.head(10)
plt.figure(figsize=(12, 6))
top_songs.plot(kind='bar', color='skyblue')
plt.title(f'Top 10 Most Popular Artists')
plt.xlabel('Artist Names')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()