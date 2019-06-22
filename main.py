from dependencies import glob, json, isfile, literal_eval
from tweet_classes import *
from nlp_utils import *
from graph_utils import *
from feature_utils import *
from classification_utils import *
from file_utils import *


def get_threads_from_story(story):
    folders = glob.glob(twitter_dataset_folder + story + '/*')
    ids = [
        folder.split("\\")[1] for folder in folders
    ]
    return folders, ids


def read_tweet_json(folder, id_thread):
    source_tweet_json_file = open(
        folder + "/source-tweet/" + id_thread + ".json", "r")
    source_tweet_json = source_tweet_json_file.read()
    source_tweet = json.loads(source_tweet_json, encoding='utf-8')
    source_tweet_json_file.close()
    reply_tweets = {}
    reply_tweets_json_files = glob.glob(folder + "/replies/*")
    for reply_tweet_json_file in reply_tweets_json_files:
        reply_thread_id = reply_tweet_json_file.split("\\")[2].split(".")[0]
        reply_tweet_json_file = open(reply_tweet_json_file, "r")
        reply_tweet_json = reply_tweet_json_file.read()
        reply_tweet = json.loads(reply_tweet_json, encoding='utf-8')
        reply_tweets[reply_thread_id] = reply_tweet
        reply_tweet_json_file.close()
    reply_structure_json_file = open(folder + "/structure.json", "r")
    reply_structure_json = reply_structure_json_file.read()
    reply_structure = json.loads(reply_structure_json, encoding='utf-8')
    reply_structure_json_file.close()
    thread = ThreadTree(id_thread, source_tweet, reply_tweets, 'data/tweets/')
    thread.build_tree(reply_structure)
    return thread


if __name__ == "__main__":

    folder_paths = ["data/class_labels", "data/datasets", "data/features/labels", "data/features/tf_idf",
                    "data/features/top", "data/features/word2vec", "data/test_tweets", "data/tf_idf_documents",
                    "data/thread_lists", "data/threads", "data/tweets", "data/vocabularies",
                    "data/word2vec_documents", "models/", "scores/ablation", "submissions/", "plots/"]

    for folder_path in folder_paths:
        if not exists(folder_path):
            makedirs(folder_path)

    twitter_dataset_folder = "../rumoureval-2019-training-data/twitter-english/"
    if not isfile('data/twitter_stories'):
        twitter_stories_folders = glob.glob(twitter_dataset_folder + "*")
        twitter_stories = [
            story_folder.split("\\")[1] for story_folder in twitter_stories_folders
        ]
        save_object(twitter_stories, 'data/twitter_stories')
    else:
        twitter_stories = load_object('data/twitter_stories')

    print('Stories: ' + str(twitter_stories))

    if not isfile('data/annotations'):
        print('Reading tweet annotations...')
        annotations_file = open(
            "../pheme-rumour-scheme-dataset/annotations/en-scheme-annotations.json", "r")
        annotations_data = [line.rstrip('\n') for line in annotations_file]
        annotation_files_veracity = glob.glob('../pheme-rumour-scheme-dataset/threads/en/*/*/annotation.json',
                                              recursive=True)

        tweet_annotations = {}
        for line in annotations_data:
            if line in ("# Source Tweets", "# Direct Replies", "# Deep Replies"):
                continue
            annotation = literal_eval(line)
            tweet_annotations[annotation['tweetid']] = annotation
        annotations_file.close()

        for file_path in annotation_files_veracity:
            tweet_id = file_path.split('\\')[2]
            annotation_file_veracity = open(file_path, 'r')
            annotation_file_veracity_data = annotation_file_veracity.read()
            annotation_file_veracity_data = re.sub('null', 'None', annotation_file_veracity_data)
            annotation = literal_eval(annotation_file_veracity_data)
            annotation_file_veracity.close()
            tweet_annotations[tweet_id].update(annotation)

        save_object(tweet_annotations, 'data/annotations')
    else:
        tweet_annotations = load_object('data/annotations')

    num_tweets = len(tweet_annotations)
    print('Total number of tweets: ' + str(num_tweets))

    ThreadTree.load_annotations_file()

    print('Processing tweet threads...')
    graph_features = {}
    for twitter_story in twitter_stories:
        if not isfile('data/thread_lists/' + twitter_story):
            thread_folders, thread_ids = get_threads_from_story(twitter_story)
            save_object((thread_folders, thread_ids), 'data/thread_lists/' + twitter_story)
        else:
            thread_folders, thread_ids = load_object('data/thread_lists/' + twitter_story)
        print('Number of threads about ' + twitter_story + ': ' + str(len(thread_ids)))
        for thread_folder, thread_id in zip(thread_folders, thread_ids):
            if not isfile('data/threads/' + thread_id):
                thread_tree = read_tweet_json(thread_folder, thread_id)
                save_object(thread_tree, 'data/threads/' + thread_id)
            else:
                thread_tree = load_object('data/threads/' + thread_id)
            if not isfile('data/features/graph_structure'):
                thread_nx_graph = thread_tree.get_nx_graph()
                graph_features.update(get_graph_features(thread_nx_graph, thread_tree))
    if not isfile('data/features/graph_structure'):
        save_object(graph_features, 'data/features/graph_structure')
    else:
        graph_features = load_object('data/features/graph_structure')

    tweet_files = glob.glob('data/tweets/*')
    vocabulary_files = (
        'data/vocabularies/tweet_unigrams', 'data/vocabularies/tweet_bigrams', 'data/vocabularies/tweet_trigrams',
        'data/vocabularies/tweet_pos_tags',
        'data/vocabularies/user_description_unigrams', 'data/vocabularies/user_description_bigrams',
        'data/vocabularies/user_description_trigrams',
        'data/vocabularies/user_description_pos_tags')
    tf_idf_files = ('data/tf_idf_documents/tweet_unigrams', 'data/tf_idf_documents/tweet_bigrams',
                    'data/tf_idf_documents/tweet_trigrams', 'data/tf_idf_documents/tweet_pos_tags',
                    'data/tf_idf_documents/user_description_unigrams',
                    'data/tf_idf_documents/user_description_bigrams',
                    'data/tf_idf_documents/user_description_trigrams',
                    'data/tf_idf_documents/user_description_pos_tags')
    if len(glob.glob('data/vocabularies/*')) < 8 or len(glob.glob('data/tf_idf_documents/*')) < 8:
        tweet_vocabularies = [[] for i in range(4)]
        user_description_vocabularies = [[] for i in range(4)]
        tweet_tf_idf_documents = [{} for i in range(4)]
        user_description_tf_idf_documents = [{} for i in range(4)]
        for i, tweet_file in enumerate(tweet_files):
            if i == 0 or (i + 1) % 250 == 0:
                print('Processing tweet #' + str(i + 1) + '...')
            tweet = load_object(tweet_file)
            if tweet.is_usable():
                tweet_id = tweet.id
                tweet_words = tweet.get_words_from_tweet_text()
                for n, ngrams in enumerate(get_ngrams_from_words(tweet_words)):
                    tweet_vocabularies[n].extend(ngrams)
                    tweet_tf_idf_documents[n][tweet_id] = ngrams
                tweet_pos_tags = get_pos_tags_for_words(tweet_words)
                tweet_vocabularies[3].extend(tweet_pos_tags)
                tweet_tf_idf_documents[3][tweet_id] = tweet_pos_tags

                user_description_words = tweet.get_words_from_user_description()
                for n, ngrams in enumerate(get_ngrams_from_words(user_description_words)):
                    user_description_vocabularies[n].extend(ngrams)
                    user_description_tf_idf_documents[n][tweet_id] = ngrams
                user_description_pos_tags = get_pos_tags_for_words(user_description_words)
                user_description_vocabularies[3].extend(user_description_pos_tags)
                user_description_tf_idf_documents[3][tweet_id] = user_description_pos_tags

        tweet_vocabularies.extend(user_description_vocabularies)
        del user_description_vocabularies
        tweet_vocabularies[3] = list(set(tweet_vocabularies[3]))
        tweet_vocabularies[7] = list(set(tweet_vocabularies[7]))
        for tweet_vocabulary, vocabulary_file in zip(tweet_vocabularies, vocabulary_files):
            save_object(tweet_vocabulary, vocabulary_file)

        tweet_tf_idf_documents.extend(user_description_tf_idf_documents)
        del user_description_tf_idf_documents
        for tweet_tf_idf_document, tf_idf_file in zip(tweet_tf_idf_documents, tf_idf_files):
            save_object(tweet_tf_idf_document, tf_idf_file)

    else:
        tweet_vocabularies = []
        for vocabulary_file in vocabulary_files:
            tweet_vocabularies.append(load_object(vocabulary_file))

        tweet_tf_idf_documents = []
        for tf_idf_file in tf_idf_files:
            tweet_tf_idf_documents.append(load_object(tf_idf_file))

    print('Total number of tokens in tweet text: ' + str(len(tweet_vocabularies[0])))
    print('Number of unique words in tweet text: ' + str(len(set(tweet_vocabularies[0]))))
    print('Total number of tokens in user description text: ' + str(len(tweet_vocabularies[4])))
    print('Number of unique words in user description text: ' + str(len(set(tweet_vocabularies[4]))))

    tf_idf_feature_files = (
        'data/features/tf_idf/tweet_unigrams', 'data/features/tf_idf/tweet_bigrams',
        'data/features/tf_idf/tweet_trigrams',
        'data/features/tf_idf/tweet_pos_tags',
        'data/features/tf_idf/user_description_unigrams', 'data/features/tf_idf/user_description_bigrams',
        'data/features/tf_idf/user_description_trigrams', 'data/features/tf_idf/user_description_pos_tags',)
    if len(glob.glob('data/features/tf_idf/*')) < 8 or not isfile('data/cleaned_vocabularies'):
        tf_idf_features = []
        cleaned_vocabularies = []
        print('Calculating TF-IDF and POS features...')
        for tf_idf_document, vocabulary, tf_idf_feature_file in zip(tweet_tf_idf_documents, tweet_vocabularies,
                                                                    tf_idf_feature_files):
            if 'pos' in tf_idf_feature_file:
                features = get_pos_features_for_documents(tf_idf_document, vocabulary)
                cleaned_vocabulary = vocabulary
            else:
                features, cleaned_vocabulary = get_tf_idf_features_for_documents(tf_idf_document, vocabulary)
            tf_idf_features.append(features)
            cleaned_vocabularies.append(cleaned_vocabulary)
            save_object(features, tf_idf_feature_file)
        save_object(cleaned_vocabularies, 'data/cleaned_vocabularies')

    else:
        tf_idf_features = []
        for tf_idf_feature_file in tf_idf_feature_files:
            tf_idf_features.append(load_object(tf_idf_feature_file))
        cleaned_vocabularies = load_object('data/cleaned_vocabularies')

    if not isfile('data/word2vec_documents/word2vec_tweet_sentences') or not isfile(
            'data/word2vec_documents/word2vec_user_description_sentences'):
        print('Splitting words into sentences...')
        word2vec_tweet_sentences = []
        word2vec_user_description_sentences = []
        tweet_sentences = {}
        user_description_sentences = {}
        for tweet_id, tweet_words in tweet_tf_idf_documents[0].items():
            sentences = get_sentences_from_words(tweet_words)
            word2vec_tweet_sentences.extend(sentences)
            tweet_sentences[tweet_id] = sentences
            user_description_words = tweet_tf_idf_documents[4][tweet_id]
            sentences = get_sentences_from_words(user_description_words)
            word2vec_user_description_sentences.extend(sentences)
            user_description_sentences[tweet_id] = sentences
        save_object(word2vec_tweet_sentences, 'data/word2vec_documents/word2vec_tweet_sentences')
        save_object(word2vec_user_description_sentences, 'data/word2vec_documents/word2vec_user_description_sentences')
        save_object(tweet_sentences, 'data/word2vec_documents/tweet_sentences')
        save_object(user_description_sentences, 'data/word2vec_documents/user_description_sentences')
    else:
        word2vec_tweet_sentences = load_object('data/word2vec_documents/word2vec_tweet_sentences')
        word2vec_user_description_sentences = load_object('data/word2vec_documents/word2vec_user_description_sentences')
        tweet_sentences = load_object('data/word2vec_documents/tweet_sentences')
        user_description_sentences = load_object('data/word2vec_documents/user_description_sentences')

    if not isfile('models/word2vec_tweets') or not isfile('models/word2vec_user_descriptions'):
        print('Training word2vec models...')
        word2vec_model_tweets = train_word2vec_model(word2vec_tweet_sentences)
        word2vec_model_tweets.save('models/word2vec_tweets')
        word2vec_model_user_descriptions = train_word2vec_model(word2vec_user_description_sentences)
        word2vec_model_user_descriptions.save('models/word2vec_user_descriptions')
    else:
        word2vec_model_tweets = word2vec.Word2Vec.load('models/word2vec_tweets')
        word2vec_model_user_descriptions = word2vec.Word2Vec.load('models/word2vec_user_descriptions')

    if not isfile('data/features/word2vec/tweets') or not isfile('data/features/word2vec/user_descriptions') \
            or not isfile('data/word2vec_documents/word_cluster_map_tweets') \
            or not isfile('data/word2vec_documents/word_cluster_map_user_descriptions'):
        print('Calculating word2vec clustering features...')
        word2vec_tweet_features, word_cluster_map_tweets = get_word2vec_word_cluster_features(word2vec_model_tweets,
                                                                                              tweet_tf_idf_documents[0])
        save_object(word2vec_tweet_features, 'data/features/word2vec/tweets')
        save_object(word_cluster_map_tweets, 'data/word2vec_documents/word_cluster_map_tweets')
        word2vec_user_description_features, word_cluster_map_user_descriptions = get_word2vec_word_cluster_features(
            word2vec_model_user_descriptions,
            tweet_tf_idf_documents[4])
        save_object(word2vec_user_description_features, 'data/features/word2vec/user_descriptions')
        save_object(word_cluster_map_user_descriptions, 'data/word2vec_documents/word_cluster_map_user_descriptions')

    else:
        word2vec_tweet_features = load_object('data/features/word2vec/tweets')
        word2vec_user_description_features = load_object('data/features/word2vec/user_descriptions')
        word_cluster_map_tweets = load_object('data/word_cluster_map_tweets')
        word_cluster_map_user_descriptions = load_object('data/word_cluster_map_user_descriptions')

    if not isfile('data/features/tweets_language_style') or not isfile(
            'data/features/user_descriptions_language_style'):
        print('Calculating language style features...')
        language_style_features_tweets = {}
        tweet_words = tweet_tf_idf_documents[0]
        user_description_words = tweet_tf_idf_documents[4]
        del tweet_tf_idf_documents
        for tweet_id, words in tweet_words.items():
            sentences = tweet_sentences[tweet_id]
            language_style_features_tweets[tweet_id] = get_extra_language_style_features(words, sentences)
        save_object(language_style_features_tweets, 'data/features/tweets_language_style')
        language_style_features_user_descriptions = {}
        for tweet_id, words in user_description_words.items():
            sentences = user_description_sentences[tweet_id]
            language_style_features_user_descriptions[tweet_id] = get_extra_language_style_features(words, sentences)
        save_object(language_style_features_user_descriptions, 'data/features/user_descriptions_language_style')
    else:
        language_style_features_tweets = load_object('data/features/tweets_language_style')
        language_style_features_user_descriptions = load_object('data/features/user_descriptions_language_style')

    if not isfile('data/features/tweet_sentiment') or not isfile('data/features/extra'):
        print('Calculating sentiment features and extra Twitter-specific features...')
        sentiment_features = {}
        extra_features = {}
        for tweet_file in tweet_files:
            tweet = load_object(tweet_file)
            if tweet.is_usable():
                tweet_id = tweet.id
                tweet_words_case_sensitive = tweet.get_words_from_tweet_text(case_sensitive=True)
                tweet_sentences_case_sensitive = get_sentences_from_words(tweet_words_case_sensitive)
                sentiment_features[tweet_id] = get_sentiment_features(tweet_sentences_case_sensitive)
                extra_features[tweet_id] = tweet.get_extra_tweet_features()
        save_object(sentiment_features, 'data/features/tweet_sentiment')
        save_object(extra_features, 'data/features/extra')

    else:
        sentiment_features = load_object('data/features/tweet_sentiment')
        extra_features = load_object('data/features/extra')

    if not isfile('data/features/labels/feature_labels'):
        feature_labels = []
        language_style_labels = ["NUMBER OF WORDS", "NUMBER OF SENTENCES", "AVERAGE WORDS PER SENTENCE",
                                 "LANGUAGE PERCENTAGE", "MEAN WORD LENGTH", "WORD LENGTH VARIANCE",
                                 "DICTIONARY PERCENTAGE", "UNIQUE WORDS PERCENTAGE"]
        feature_labels.extend(language_style_labels)
        feature_labels.extend(["[USER DESCRIPTION] " + label for label in language_style_labels])
        for vocabulary in cleaned_vocabularies[0:3]:
            feature_labels.extend([ngram_tuple_to_string(ngram) + " TF-IDF" for ngram, _ in vocabulary])
        feature_labels.extend([pos_tag + " TF-IDF" for pos_tag in cleaned_vocabularies[3]])
        for vocabulary in cleaned_vocabularies[4:7]:
            feature_labels.extend(
                ["[USER DESCRIPTION] " + ngram_tuple_to_string(ngram) + " TF-IDF" for ngram, _ in
                 vocabulary])
        feature_labels.extend(["[USER DESCRIPTION] " + pos_tag + " TF-IDF" for pos_tag in cleaned_vocabularies[7]])
        num_word2vec_clusters = len(list(word2vec_tweet_features.values())[0])
        feature_labels.extend(["WORD2VEC CLUSTER " + str(i + 1) for i in range(num_word2vec_clusters)])
        num_word2vec_clusters = len(list(word2vec_user_description_features.values())[0])
        feature_labels.extend(
            ["[USER DESCRIPTION] WORD2VEC CLUSTER " + str(i + 1) for i in range(num_word2vec_clusters)])
        feature_labels.extend(["MEAN POSITIVE SENTIMENT", "MEAN NEUTRAL SENTIMENT", "MEAN NEGATIVE SENTIMENT",
                               "POSITIVE SENTIMENT VARIANCE", "NEUTRAL SENTIMENT VARIANCE",
                               "NEGATIVE SENTIMENT VARIANCE"])
        feature_labels.extend(
            ["DFS PRIORITY", "BFS PRIORITY", "DEGREE CENTRALITY", "BETWEENNESS CENTRALITY", "CLOSENESS CENTRALITY",
             "HUB SCORE", "PAGERANK"])
        feature_labels.extend(['EVIDENTIALITY', 'CERTAINTY', 'CHARACTER COUNT', 'FAVORITE COUNT', 'RETWEET COUNT',
                               'AGE', 'USER IS VERIFIED', 'USER FOLLOWERS COUNT', 'USER STATUSES COUNT',
                               'USER FRIENDS COUNT', 'USER FAVORITES COUNT', 'USER LISTED COUNT', 'USER PROFILE AGE',
                               'USER USES A BACKGROUD IMAGE', 'USER HAS THE DEFAULT PROFILE IMAGE',
                               'USER PROFILE TEXT COLOR', 'USER PROFILE SIDEBAR FILL COLOR',
                               'USER PROFILE SIDEBAR BORDER COLOR', 'USER PROFILE BACKGROUND COLOR',
                               'USER PROFILE LINK COLOR'])
        save_object(feature_labels, 'data/features/labels/feature_labels')
    else:
        feature_labels = load_object('data/features/labels/feature_labels')

    num_features = len(feature_labels)
    print('Total number of features: ' + str(num_features))

    del tweet_vocabularies, cleaned_vocabularies

    if not isfile('data/datasets/dataset_dictionary'):
        dataset = {}
        tweet_ids = list(extra_features.keys())
        for tweet_id in tweet_ids:
            dataset[tweet_id] = language_style_features_tweets[tweet_id]
            dataset[tweet_id].extend(language_style_features_user_descriptions[tweet_id])
            for tf_idf_feature_set in tf_idf_features:
                dataset[tweet_id].extend(tf_idf_feature_set[tweet_id])
            dataset[tweet_id].extend(word2vec_tweet_features[tweet_id])
            dataset[tweet_id].extend(word2vec_user_description_features[tweet_id])
            dataset[tweet_id].extend(sentiment_features[tweet_id])
            dataset[tweet_id].extend(graph_features[tweet_id])
            dataset[tweet_id].extend(extra_features[tweet_id])
        save_object(dataset, 'data/datasets/dataset_dictionary')
        del language_style_features_tweets, language_style_features_user_descriptions, tf_idf_features, \
            word2vec_tweet_features, word2vec_user_description_features, sentiment_features, \
            graph_features, extra_features
    else:
        del language_style_features_tweets, language_style_features_user_descriptions, tf_idf_features, \
            word2vec_tweet_features, word2vec_user_description_features, sentiment_features, \
            graph_features, extra_features
        dataset = load_object('data/datasets/dataset_dictionary')

    if not isfile('data/datasets/dataset_task_a.tsv') or not isfile('data/class_labels/class_labels_task_a') \
            or not isfile('data/datasets/dataset_task_b.tsv') or not isfile('data/class_labels/class_labels_task_b') \
            or not isfile('data/features/labels/feature_labels_task_a') \
            or not isfile('data/features/labels/feature_labels_task_b'):
        class_labels_task_a = []
        dataset_task_a = []
        feature_labels_task_a = list(feature_labels)
        feature_labels_task_a.extend([label + " DIFFERENCE" for label in feature_labels])
        class_labels_task_b = []
        dataset_task_b = []
        feature_labels_task_b = feature_labels
        for tweet_file in tweet_files:
            tweet = load_object(tweet_file)
            if tweet.is_usable() and tweet.annotation is not None:
                tweet_id = tweet.id
                if 'responsetype-vs-source' in tweet.annotation.keys() and tweet.source_tweet is not None \
                        and tweet.source_tweet.is_usable():
                    class_labels_task_a.append(get_task_a_label(tweet.annotation['responsetype-vs-source']))
                    source_tweet_id = tweet.source_tweet.id
                    task_a_features_for_tweet = dataset[tweet_id]
                    task_a_features_for_tweet.extend(
                        difference_of_feature_vectors(dataset[tweet_id], dataset[source_tweet_id]))
                    dataset_task_a.append(task_a_features_for_tweet)
                if 'true' in tweet.annotation.keys():
                    class_labels_task_b.append(int(tweet.annotation['true']))
                    dataset_task_b.append(dataset[tweet_id])

        del dataset
        save_object(class_labels_task_a, 'data/class_labels/class_labels_task_a')
        dataset_task_a = pd.DataFrame(dataset_task_a, columns=feature_labels_task_a)
        save_object(feature_labels_task_a, 'data/features/labels/feature_labels_task_a')
        dataset_task_a.to_csv('data/datasets/dataset_task_a.tsv', sep='\t', header=True, index=False, encoding='utf-8')

        save_object(class_labels_task_b, 'data/class_labels/class_labels_task_b')
        dataset_task_b = pd.DataFrame(dataset_task_b, columns=feature_labels_task_b)
        save_object(feature_labels_task_b, 'data/features/labels/feature_labels_task_b')
        dataset_task_b.to_csv('data/datasets/dataset_task_b.tsv', sep='\t', header=True, index=False, encoding='utf-8')

    else:
        del dataset
        class_labels_task_a = load_object('data/class_labels/class_labels_task_a')
        dataset_task_a = pd.read_csv('data/datasets/dataset_task_a.tsv', sep='\t', index_col=False, header=0,
                                     encoding='utf-8')
        feature_labels_task_a = load_object('data/features/labels/feature_labels_task_a')

        class_labels_task_b = load_object('data/class_labels/class_labels_task_b')
        dataset_task_b = pd.read_csv('data/datasets/dataset_task_b.tsv', sep='\t', index_col=False, header=0,
                                     encoding='utf-8')
        feature_labels_task_b = load_object('data/features/labels/feature_labels_task_b')

    print('Class distribution for Task A: ' + str(get_class_distribution(class_labels_task_a)))
    print('Splitting dataset for One-vs-Rest classification...')
    dataset_tuple_1, dataset_tuple_2, dataset_tuple_3 = split_dataset_task_a_one_vs_rest(dataset_task_a,
                                                                                         class_labels_task_a)
    del dataset_task_a, class_labels_task_a
    evaluation_metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    optimize_parameters = False
    if not isfile('models/task_a_comment_model') or not isfile('scores/task_a_comment_model_scores.tsv'):
        if not isfile('data/datasets/task_a_comment_dataset.tsv') \
                or not isfile('data/class_labels/task_a_comment_class_labels') \
                or not isfile('data/features/labels/task_a_comment_feature_labels'):
            comment_dataset, comment_labels = dataset_tuple_1
            comment_labels = get_task_a_binary_labels(comment_labels)
            print('Original dataset size: ' + str(comment_dataset.shape))
            print('Performing feature selection on dataset...')
            feature_selector = FeatureSelector(comment_dataset, comment_labels)
            comment_dataset, comment_feature_labels = feature_selector.perform_whole_pipeline()
            save_object(feature_selector.get_top_n_features(20), 'data/features/top/top_comment_features')
            del feature_selector
            print('New reduced number of features: ' + str(len(comment_feature_labels)))
            comment_dataset.to_csv('data/datasets/task_a_comment_dataset.tsv', sep='\t', header=True, index=False,
                                   encoding='utf-8')
            save_object(comment_labels, 'data/class_labels/task_a_comment_class_labels')
            save_object(comment_feature_labels, 'data/features/labels/task_a_comment_feature_labels')

        else:
            comment_dataset = pd.read_csv('data/datasets/task_a_comment_dataset.tsv', sep='\t', index_col=False,
                                          header=0,
                                          encoding='utf-8')
            comment_labels = load_object('data/class_labels/task_a_comment_class_labels')

        print("Training comment model for Task A...")
        comment_model, scores = train_model(comment_dataset, comment_labels, optimize_parameters=optimize_parameters)
        print(scores)
        save_object(comment_model, 'models/task_a_comment_model')
        scores = pd.DataFrame(np.array(list(scores.values()), dtype='float'), index=list(scores.keys()),
                              columns=evaluation_metrics)
        scores.to_csv('scores/task_a_comment_model_scores.tsv', sep='\t', index=True, header=True, encoding='utf-8')
    else:
        comment_model = load_object('models/task_a_comment_model')

    if not isfile('models/task_a_query_model') or not isfile('scores/task_a_query_model_scores.tsv'):
        if not isfile('data/datasets/task_a_query_dataset.tsv') \
                or not isfile('data/class_labels/task_a_query_class_labels') \
                or not isfile('data/features/labels/task_a_query_feature_labels'):
            query_dataset, query_labels = dataset_tuple_2
            query_labels = get_task_a_binary_labels(query_labels)
            print('Original dataset size: ' + str(query_dataset.shape))
            print('Performing feature selection on dataset...')
            feature_selector = FeatureSelector(query_dataset, query_labels)
            query_dataset, query_feature_labels = feature_selector.perform_whole_pipeline()
            save_object(feature_selector.get_top_n_features(20), 'data/features/top/top_query_features')
            del feature_selector
            print('New reduced number of features: ' + str(len(query_feature_labels)))
            query_dataset.to_csv('data/datasets/task_a_query_dataset.tsv', sep='\t', header=True, index=False,
                                 encoding='utf-8')
            save_object(query_labels, 'data/class_labels/task_a_query_class_labels')
            save_object(query_feature_labels, 'data/features/labels/task_a_query_feature_labels')

        else:
            query_dataset = pd.read_csv('data/datasets/task_a_query_dataset.tsv', sep='\t', index_col=False, header=0,
                                        encoding='utf-8')
            query_labels = load_object('data/class_labels/task_a_query_class_labels')
        print("Training query model for Task A...")
        query_model, scores = train_model(query_dataset, query_labels, optimize_parameters=optimize_parameters)
        print(scores)
        save_object(query_model, 'models/task_a_query_model')
        scores = pd.DataFrame(np.array(list(scores.values()), dtype='float'), index=list(scores.keys()),
                              columns=evaluation_metrics)
        scores.to_csv('scores/task_a_query_model_scores.tsv', sep='\t', index=True, header=True, encoding='utf-8')
    else:
        query_model = load_object('models/task_a_query_model')
    if not isfile('models/task_a_support_deny_model') or not isfile('models/task_a_support_deny_model_scores.tsv'):
        if not isfile('data/datasets/task_a_support_deny_dataset.tsv') \
                or not isfile('data/class_labels/task_a_support_deny_class_labels') \
                or not isfile('data/features/labels/task_a_support_deny_feature_labels'):
            support_deny_dataset, support_deny_labels = dataset_tuple_3
            support_deny_labels = get_task_a_binary_labels(support_deny_labels)
            print('Original dataset size: ' + str(support_deny_dataset.shape))
            print('Performing feature selection on dataset...')
            feature_selector = FeatureSelector(support_deny_dataset, support_deny_labels)
            support_deny_dataset, support_deny_feature_labels = feature_selector.perform_whole_pipeline()
            save_object(feature_selector.get_top_n_features(20), 'data/features/top/top_support_deny_features')
            del feature_selector
            print('New reduced number of features: ' + str(len(support_deny_feature_labels)))
            support_deny_dataset.to_csv('data/datasets/task_a_support_deny_dataset.tsv', sep='\t', header=True,
                                        index=False,
                                        encoding='utf-8')
            save_object(support_deny_labels, 'data/class_labels/task_a_support_deny_class_labels')
            save_object(support_deny_feature_labels, 'data/features/labels/task_a_support_deny_feature_labels')

        else:
            support_deny_dataset = pd.read_csv('data/datasets/task_a_support_deny_dataset.tsv', sep='\t',
                                               index_col=False,
                                               header=0,
                                               encoding='utf-8')
            support_deny_labels = load_object('data/class_labels/task_a_support_deny_class_labels')
        print("Training support vs deny model for Task A...")
        support_deny_model, scores = train_model(support_deny_dataset, support_deny_labels,
                                                 optimize_parameters=optimize_parameters)
        print(scores)
        save_object(support_deny_model, 'models/task_a_support_deny_model')
        scores = pd.DataFrame(np.array(list(scores.values()), dtype='float'), index=list(scores.keys()),
                              columns=evaluation_metrics)
        scores.to_csv('scores/task_a_support_deny_model_scores.tsv', sep='\t', index=True, header=True,
                      encoding='utf-8')
    else:
        support_deny_model = load_object('models/task_a_support_deny_model')

    evaluation_metrics.append('Cross-entropy loss')
    print('Class distribution for Task B: ' + str(get_class_distribution(class_labels_task_b)))
    if not isfile('models/task_b_veracity_model') or not isfile('scores/task_b_veracity_model_scores.tsv'):
        if not isfile('data/features/labels/task_b_feature_labels'):
            print('Original dataset size: ' + str(dataset_task_b.shape))
            print('Performing feature selection on dataset...')
            feature_selector = FeatureSelector(dataset_task_b, class_labels_task_b)
            dataset_task_b, task_b_feature_labels = feature_selector.perform_whole_pipeline()
            save_object(feature_selector.get_top_n_features(20), 'data/features/top/top_veracity_features')
            del feature_selector
            print('New reduced number of features: ' + str(len(task_b_feature_labels)))
            save_object(task_b_feature_labels, 'data/features/labels/task_b_feature_labels')
            dataset_task_b.to_csv('data/datasets/dataset_task_b.tsv', sep='\t', header=True, index=False,
                                  encoding='utf-8')

        print("Training veracity model for Task B...")
        veracity_model, scores = train_model(dataset_task_b, class_labels_task_b, balance_dataset=False,
                                             classify_with_probabilities=True,
                                             optimize_parameters=optimize_parameters)
        print(scores)
        save_object(veracity_model, 'models/task_b_veracity_model')
        scores = pd.DataFrame(np.array(list(scores.values()), dtype='float'), index=list(scores.keys()),
                              columns=evaluation_metrics)
        scores.to_csv('scores/task_b_veracity_model_scores.tsv', sep='\t', index=True, header=True, encoding='utf-8')
    else:
        veracity_model = load_object('models/task_b_veracity_model')
