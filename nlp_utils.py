from dependencies import np, re, logging, download, stopwords, nltk_dictionary, wordnet, WordNetLemmatizer, \
    SentimentIntensityAnalyzer, pos_tag, ngrams, TextCollection, FreqDist, word2vec, KMeans

download("words")
download('stopwords')
download('wordnet')
download('vader_lexicon')

stops = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

special_tokens = ('QUOTE', 'EMAIL', 'USER', 'NUM', 'HASHTAG', '.', ',', '?', '!', ':', '-', '+', '\n', '\r')

sia = SentimentIntensityAnalyzer()

nltk_dictionary_word_set = set(nltk_dictionary.words())


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def post_process_words(words):
    processed = []
    for word in words:
        if word not in special_tokens and word not in stops:
            word_pos_tag = pos_tag([word, ])[0][1]
            processed.append(lemmatizer.lemmatize(word, get_wordnet_pos(word_pos_tag)))
        elif word in special_tokens:
            processed.append(word)

    return processed


def get_sentences_from_words(words):
    sentences = []
    sentence = []
    in_sentence = False
    for word in words:
        if word in ('.', '?', '!', '\n', '\r'):
            if in_sentence:
                sentence.append(word)
                sentences.append(sentence)
            in_sentence = False
            sentence = []
        else:
            in_sentence = True
            sentence.append(word)
    if in_sentence:
        sentences.append(sentence)
    return sentences


def get_ngrams_from_words(words):
    unigrams = list(words)
    words = ['START', ]
    words.extend(unigrams)
    words.append('END')
    bigrams = list(ngrams(words, 2))
    trigrams = list(ngrams(words, 3))
    return unigrams, bigrams, trigrams


def ngram_tuple_to_string(ngram):
    if type(ngram) is tuple:
        string = " ".join(ngram)
    else:
        string = ngram
    string = re.sub('\n', 'NEWLINE', string)
    string = re.sub('\r', 'RETURN', string)
    string = re.sub('\t', 'TAB', string)
    return string


def get_extra_language_style_features(words, sentences):
    features = []
    words_filtered = [word for word in words if word not in special_tokens]
    num_words = len(words_filtered)
    num_sentences = len(sentences)
    if num_words > 0:
        features.append(num_words)
        features.append(num_sentences)
        features.append(num_words / num_sentences)
        features.append(num_words / len(words))
        word_lengths = [len(word) for word in words_filtered]
        features.append(np.mean(word_lengths))
        features.append(np.var(word_lengths))
        non_dictionary_words = [word for word in words_filtered if word not in nltk_dictionary_word_set]
        features.append(1 - (len(non_dictionary_words) / num_words))
        unique_words = set(words_filtered)
        features.append(len(unique_words) / num_words)
    else:
        features = [0, ] * 8
        features[1] = num_sentences
    return features


def get_tf_idf_features_for_documents(documents, vocabulary):
    documents_words = list(documents.values())
    collection = TextCollection(documents_words)
    tokens_freq = FreqDist(vocabulary)
    if type(documents_words[0][0]) is str:
        n_gram_length = 1
    else:
        n_gram_length = len(documents_words[0][0])
    fractions = [8, 16, 20]
    num_most_common = len(set(vocabulary)) // fractions[n_gram_length - 1]
    vocabulary_cleaned = tokens_freq.most_common(num_most_common)
    features = {}
    for tweet_id, document in documents.items():
        if len(document) == 0:
            features[tweet_id] = [0, ] * num_most_common
        else:
            features[tweet_id] = [collection.tf_idf(word, document) if word in document else 0
                                  for word, _ in vocabulary_cleaned]
    return features, vocabulary_cleaned


def get_tf_idf_features_for_test_documents(documents, vocabulary):
    documents_words = list(documents.values())
    collection = TextCollection(documents_words)
    features = {}
    vocabulary_size = len(vocabulary)
    for tweet_id, document in documents.items():
        if len(document) == 0:
            features[tweet_id] = [0, ] * vocabulary_size
        else:
            features[tweet_id] = [collection.tf_idf(word, document) if word in document else 0
                                  for word, _ in vocabulary]
    return features


def get_pos_tags_for_words(words):
    words_filtered = [word for word in words if word not in special_tokens]
    pos_tags = pos_tag(words_filtered)
    pos_tags = [tag for _, tag in pos_tags]
    return pos_tags


def get_pos_features_for_documents(documents, vocabulary):
    documents_words = list(documents.values())
    collection = TextCollection(documents_words)
    features = {}
    vocabulary_size = len(vocabulary)
    for tweet_id, document in documents.items():
        if len(document) == 0:
            features[tweet_id] = [0, ] * vocabulary_size
        else:
            features[tweet_id] = [collection.tf_idf(tag, document) if tag in document else 0 for tag in vocabulary]
    return features


def train_word2vec_model(sentences):
    print('Removing stop-words and lemmatizing sentences...')
    for i, sentence in enumerate(sentences):
        sentences[i] = post_process_words(sentence)

    num_features = 500  # Word vector dimensionality
    min_word_count = 10  # Minimum word count
    num_workers = 6  # Number of threads to run in parallel
    context = 5  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    model = word2vec.Word2Vec(
        sentences,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context,
        sample=downsampling,
        seed=57)
    word2vec.Word2Vec()
    model.init_sims(replace=True)

    return model


def get_word2vec_word_cluster_features(model, documents):
    word2vec_vocab = list(model.wv.index2word)
    word2vec_vocab_size = len(word2vec_vocab)
    print('word2vec vocabulary size: ' + str(word2vec_vocab_size))
    k_means = KMeans(
        n_clusters=word2vec_vocab_size // 10,
        n_init=5,
        max_iter=1000,
        tol=1e-6,
        random_state=57,
        n_jobs=-1)
    cluster_index = k_means.fit_predict(model.trainables.syn1neg)
    word_centroid_map = dict(zip(word2vec_vocab, cluster_index))
    num_centroids = max(word_centroid_map.values()) + 1
    print('Number of word clusters: ' + str(num_centroids))
    features = {}
    for tweet_id, document in documents.items():
        document = post_process_words(document)
        bag_of_centroids = [0, ] * num_centroids
        num_words = len(document)
        for word in document:
            if word in word2vec_vocab:
                bag_of_centroids[word_centroid_map[word]] += (1 / num_words)
        features[tweet_id] = bag_of_centroids
    return features, word_centroid_map


def get_word2vec_word_cluster_features_for_test_documents(word_cluster_map, documents):
    num_centroids = max(word_cluster_map.values()) + 1
    word2vec_vocab = list(word_cluster_map.keys())
    features = {}
    for tweet_id, document in documents.items():
        document = post_process_words(document)
        bag_of_centroids = [0, ] * num_centroids
        num_words = len(document)
        for word in document:
            if word in word2vec_vocab:
                bag_of_centroids[word_cluster_map[word]] += (1 / num_words)
        features[tweet_id] = bag_of_centroids
    return features


def get_words_in_word2vec_cluster(word_cluster_map, cluster_id):
    return [word for word, cluster in word_cluster_map.items() if cluster == cluster_id]


def get_sentiment_features(sentences):
    sentiment_scores = []
    for sentence in sentences:
        sentence_concat = ' '.join(sentence)
        sentence_concat = re.sub(r' \'', '\'', sentence_concat)
        sentence_concat = re.sub(r' \.', '.', sentence_concat)
        sentence_concat = re.sub(' ,', ',', sentence_concat)
        sentence_concat = re.sub(r' \?', '?', sentence_concat)
        sentence_concat = re.sub(' !', '!', sentence_concat)
        sentence_concat = re.sub(' :', ':', sentence_concat)
        sentence_concat = re.sub(' - ', '-', sentence_concat)
        sentence_concat = re.sub(r' \+ ', '+', sentence_concat)
        sentence_concat = re.sub(' \n', '', sentence_concat)
        sentence_concat = re.sub(' \r', '', sentence_concat)
        scores = sia.polarity_scores(sentence_concat)
        sentiment_scores.append([scores['pos'], scores['neu'], scores['neg']])
    features = []
    features.extend(np.mean(sentiment_scores, axis=0))
    features.extend(np.var(sentiment_scores, axis=0))
    return features
