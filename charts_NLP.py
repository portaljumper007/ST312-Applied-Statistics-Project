import pandas as pd
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm  # Import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
def filter_by_sentiment_intensity(tokens):
    return [word for word in tokens if abs(sid.polarity_scores(word)['compound']) > 0.6]

def augment_corpus_with_custom_keywords(processed_texts, keyword, boost_factor=100):
    augmented_texts = []
    for text in processed_texts:
        if keyword in text:
            augmented_texts.append(text + [keyword] * boost_factor)
        else:
            augmented_texts.append(text)
    return augmented_texts

def main(show_graph=True, num_topics=5):

    print("Loading dataset...")
    music_df = pd.read_csv("Music Charts.csv")
    music_df['WeekID'] = pd.to_datetime(music_df['WeekID'])

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Load stopwords and lemmatizer just once
    custom_stop_words = ['like','know','time','one','na','let','go','new','want','back','come','take'] # Adding our own stopwords because nltk ones aren't enough
    stop_words = set(stopwords.words('english') + custom_stop_words)
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        tokens = nltk.word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [word for word in tokens if len(word) > 3]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        tokens = filter_by_sentiment_intensity(tokens)
        return tokens

    print("Preprocessing text...")
    music_df['text'] = music_df['Song']# + " " + music_df['Performer']
    chunk_size = 1000
    processed_texts = []
    for start in tqdm(range(0, len(music_df), chunk_size), desc="Processing Text"):
        chunk = music_df['text'][start:start + chunk_size]
        processed_chunk = chunk.apply(preprocess_text)
        processed_texts.extend(processed_chunk)

    custom_keyword = "christmas"
    processed_texts = augment_corpus_with_custom_keywords(processed_texts, custom_keyword)

    music_df['text'] = processed_texts

    print("Creating dictionary and document-term matrix...")
    dictionary = corpora.Dictionary(music_df['text'])
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in music_df['text']]

    print("Creating LDA model...")
    num_topics = 5
    num_threads = 12
    lda = models.ldamulticore.LdaMulticore(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=5, workers=num_threads)  # Using LdaMulticore

    print("Analyzing topics across weeks...")
    weekly_topic_strengths = defaultdict(lambda: defaultdict(int))

    for index, row in tqdm(music_df.iterrows(), total=music_df.shape[0], desc="Analyzing Topics"):
        week = row['WeekID']
        position = row['Week Position']
        topics = lda.get_document_topics(doc_term_matrix[index])

        for topic, strength in topics:
            weighted_strength = strength / position
            weekly_topic_strengths[week][topic] += weighted_strength

    if show_graph:
        # Extracting topic descriptions
        topic_descriptions = {i: ' '.join([word for word, prob in lda.show_topic(i, topn=5) if prob > 0.01]) for i in range(num_topics)}

        print("Plotting topic strengths over time...")
        plt.figure(figsize=(15, 10))

        for topic in range(num_topics):
            weeks = sorted(weekly_topic_strengths.keys())
            strengths = [weekly_topic_strengths[week][topic] for week in weeks]
            topic_label = f'Topic {topic + 1}: {topic_descriptions[topic]}'
            plt.plot(weeks, strengths, label=topic_label)

        plt.title("Topic Strengths Over Time in Billboard Music Charts")
        plt.xlabel("Week")
        plt.ylabel("Topic Strength")
        plt.legend()
        plt.show()
    return weekly_topic_strengths


if __name__ == '__main__': #Check we're in the main thread for multiprocessing inside gensim multicore LDA
    main(show_graph=True)

