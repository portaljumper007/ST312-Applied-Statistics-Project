import pandas as pd
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from tqdm import tqdm  # Progress bar library
from nltk.sentiment import SentimentIntensityAnalyzer
import pickle
from sklearn.cluster import KMeans
import numpy as np
import scipy.stats as stats
import plotly.graph_objs as go
import os
from plotly.subplots import make_subplots
from collections import Counter
import random

sid = SentimentIntensityAnalyzer()

# Seed all randomness
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def filter_by_sentiment_intensity(tokens):
    return [word for word in tokens if abs(sid.polarity_scores(word)['compound']) > 0.025]

def augment_corpus_with_custom_keywords(processed_texts, keyword, boost_factor=100):
    augmented_texts = []
    for text in processed_texts:
        if keyword in text:
            augmented_texts.append(text + [keyword] * boost_factor)
        else:
            augmented_texts.append(text)
    return augmented_texts

def make_default_dict():
    return defaultdict(int)

def load_or_create_data(dictionary_filename, weekly_strengths_filename, lda_model_filename, num_topics):
    if os.path.exists(dictionary_filename) and os.path.exists(weekly_strengths_filename) and os.path.exists(lda_model_filename):
        with open(dictionary_filename, 'rb') as file:
            dictionary = pickle.load(file)
        with open(weekly_strengths_filename, 'rb') as file:
            weekly_topic_strengths = pickle.load(file)
        with open(lda_model_filename, 'rb') as file:
            lda = pickle.load(file)
    else:
        print("Loading dataset...")
        music_df = pd.read_csv("Music Charts.csv")
        music_df['WeekID'] = pd.to_datetime(music_df['WeekID'])

        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        # Load stopwords and lemmatizer just once
        custom_stop_words = ['like', 'know', 'time', 'one', 'na', 'let', 'go', 'new', 'want', 'back', 'come', 'take']  
        stop_words = set(stopwords.words('english'))# + custom_stop_words)
        lemmatizer = WordNetLemmatizer()

        def preprocess_text(text):
            tokens = nltk.word_tokenize(text.lower())
            tokens = [word for word in tokens if word.isalpha()]
            tokens = [word for word in tokens if word not in stop_words]
            tokens = [word for word in tokens if len(word) > 2]
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            tokens = filter_by_sentiment_intensity(tokens)
            return tokens

        print("Preprocessing text...")
        
        music_df['text'] = music_df['Song']  
        chunk_size = 10000
        processed_texts = []
        for start in tqdm(range(0, len(music_df), chunk_size), desc="Processing Text"):
            chunk = music_df['text'][start:start + chunk_size]
            processed_chunk = chunk.apply(preprocess_text)
            processed_texts.extend(processed_chunk)

        #custom_keyword = "christmas"
        #processed_texts = augment_corpus_with_custom_keywords(processed_texts, custom_keyword)
        music_df['text'] = processed_texts


        all_words = [word for text in processed_texts for word in text]
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(20)

        words, frequencies = zip(*top_words)

        fig = go.Figure(data=[go.Bar(x=words, y=frequencies)])
        fig.update_layout(title='Top 20 Most Frequent Words in Song Titles (After Preprocessing)',
                        xaxis_title='Word',
                        yaxis_title='Frequency',
                        xaxis_tickangle=-45)
        fig.show()


        print("Creating dictionary and document-term matrix...")
        dictionary = corpora.Dictionary(processed_texts)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_texts]

        print("Creating LDA model...")
        num_threads = 12
        #documentation for LDA multicore: https://radimrehurek.com/gensim/models/ldamulticore.html
        lda = models.ldamulticore.LdaMulticore(doc_term_matrix, num_topics=60, id2word=dictionary, passes=10, workers=num_threads, alpha='asymmetric', eta=0.1, random_state=42) #this is still somewhat random due to "non-determinism in OS scheduling of the worker processes"

        print("Analyzing topics across weeks...")
        weekly_topic_strengths = defaultdict(make_default_dict)

        for index, row in tqdm(music_df.iterrows(), total=music_df.shape[0], desc="Analyzing Topics"):
            week = row['WeekID']
            position = row['Week Position']
            topics = lda.get_document_topics(doc_term_matrix[index])

            for topic, strength in topics:
                weighted_strength = strength / position
                weekly_topic_strengths[week][topic] += weighted_strength

        with open(dictionary_filename, 'wb') as file:
            pickle.dump(dictionary, file)
        with open(weekly_strengths_filename, 'wb') as file:
            pickle.dump(weekly_topic_strengths, file)
        with open(lda_model_filename, 'wb') as file:
            pickle.dump(lda, file)

    return dictionary, weekly_topic_strengths, lda

def main(show_graph=True, num_topics=10, specific_year=None):
    dictionary_filename = 'dictionary.pkl'
    weekly_strengths_filename = 'weekly_topic_strengths.pkl'
    lda_model_filename = 'lda_model.pkl'
    dictionary, weekly_topic_strengths, lda = load_or_create_data(dictionary_filename, weekly_strengths_filename, lda_model_filename, num_topics)


    # Cluster topics based on their word distributions
    topic_words = np.zeros((len(lda.get_topics()), len(dictionary)))  # All topics, all words
    for topic in range(len(lda.get_topics())):
        word_strength = lda.get_topic_terms(topic)  # Get all words for the topic
        for word_id, strength in word_strength:
            topic_words[topic, word_id] = strength
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=60)  # Choose an appropriate number of components
    topic_words_reduced = pca.fit_transform(topic_words)

    from sklearn.metrics.pairwise import cosine_distances
    kmeans = KMeans(n_clusters=num_topics, init='k-means++', n_init=200, max_iter=300, random_state=42)
    topic_words_cosine_dist = cosine_distances(topic_words_reduced)
    kmeans.fit(topic_words_cosine_dist)


    topic_clusters = defaultdict(list)
    for topic, cluster_label in enumerate(kmeans.labels_):
        topic_clusters[cluster_label].append(topic)
    print(topic_clusters)

    # Select representative topics from each cluster based on weight distribution and variance of strengths
    selected_topics = []
    for cluster_topics in topic_clusters.values():
        cluster_topic_scores = []
        for topic in cluster_topics:
            # Calculate weight distribution metric
            topic_words = lda.show_topic(topic, topn=10)
            _, probs = zip(*topic_words)
            weight_distribution = np.std(probs)  # Higher std = more evenly distributed weights
            gini_impurity = 1 - np.sum(np.array(probs)**2)
            # Calculate variance of topic strengths
            topic_strengths = [weekly_topic_strengths[week].get(topic, 0) for week in weekly_topic_strengths.keys()]
            strength_variance = np.var(topic_strengths)
            # Combine weight distribution and variance scores
            combined_score = gini_impurity * strength_variance# / np.sum(topic_strengths)
            cluster_topic_scores.append((topic, combined_score))

        selected_topic = max(cluster_topic_scores, key=lambda x: x[1])[0]
        selected_topics.append(selected_topic)

    print("Selected topics...", selected_topics)

    # Extracting topic descriptions and saving top words and probabilities for selected topics
    topic_info = {selected_topics.index(i): lda.show_topic(i, topn=10) for i in selected_topics}
    with open('topic_info.pkl', 'wb') as file:
        pickle.dump(topic_info, file)

    if show_graph:
        topic_descriptions = {i: ' '.join([word for word, _ in topic_info[i][:10]]) for i in range(len(selected_topics))}
        weekly_topic_strengths_norm = defaultdict(dict)

        print("BAM", weekly_topic_strengths[list(weekly_topic_strengths.keys())[0]])

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Topic Strengths Over Time (Raw)", "Topic Strengths Over Time (Rank Normalized)"))
        print("Plotting topic strengths over time...")
        # Plot raw topic strengths
        j = -1
        for topic in selected_topics:
            j += 1
            weeks = sorted(weekly_topic_strengths.keys())
            strengths = [weekly_topic_strengths[week][topic] for week in weeks]
            topic_label = f'Topic {topic + 1}: {topic_descriptions[selected_topics.index(topic)]}'
            fig.add_trace(go.Scatter(x=weeks, y=strengths, mode='lines', name=topic_label), row=1, col=1)
        j = -1
        for topic in selected_topics:
            j += 1
            weeks = sorted(weekly_topic_strengths.keys())
            # Apply rank normalization
            topic_strengths = [weekly_topic_strengths[week][j] for week in weeks]
            ranks = stats.rankdata(topic_strengths)
            unique_ranks, inverse_indices = np.unique(ranks, return_inverse=True)
            quantiles = (ranks - 1) / (len(ranks) - 1)

            if False:
                # Store normalized strengths in weekly_topic_strengths_norm
                for week, strength in zip(weeks, topic_strengths):
                    rank_index = inverse_indices[topic_strengths.index(strength)]
                    quantile = quantiles[rank_index]
                    weekly_topic_strengths_norm[week][j] = quantile
            
            weekly_topic_strengths_norm = weekly_topic_strengths

            topic_label = f'Topic {topic + 1} (0-10):{j + 1}: {topic_descriptions[j]}'
            fig.add_trace(go.Scatter(x=weeks, y=[weekly_topic_strengths_norm[week][j] for week in weeks], mode='lines', name=topic_label), row=2, col=1)


        fig.update_layout(height=800, xaxis1_title="Week", yaxis1_title="Topic Strength", xaxis2_title="Week", yaxis2_title="Topic Strength (Rank Normalized)")
        if specific_year is not None:
            fig.update_layout(title_text=f"Topic Strengths Over Time in Billboard Music Charts - {specific_year}")
        else:
            fig.update_layout(title_text="Topic Strengths Over Time in Billboard Music Charts")
        fig.show()


        # Plot the topic strength matrix
        import plotly.figure_factory as ff
        print("Plotting topic strength matrix heatmap...")
        z = []
        annotations = []
        for topic, top_words in topic_info.items():
            words, probs = zip(*top_words)
            z.append(list(probs))
            annotations.append([f"{word}<br>{prob:.4f}" for word, prob in zip(words, probs)])
        x = list(range(10))  # x-axis labels (1st, 2nd, ..., 10th)
        y = [f"Topic {i+1}" for i in range(num_topics)]  # y-axis labels (Topic 1, Topic 2, ...)

        # Create the annotated heatmap
        word_strength_matrix = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=annotations, colorscale='blues', zmid=0)
        word_strength_matrix.update_layout(title='Top Words and Strengths for Each Topic', height=1200, width=1200, 
                                        xaxis=dict(tickvals=np.arange(10), ticktext=['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th']),
                                        yaxis=dict(tickvals=[i for i in range(num_topics)], ticktext=[f"Topic {i+1}" for i in range(num_topics)]),
                                        showlegend=False)

        word_strength_matrix.show()

    return weekly_topic_strengths_norm

if __name__ == '__main__':
    main(show_graph=True, num_topics=10, specific_year=None)