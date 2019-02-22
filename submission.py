from dependencies import glob, json, isfile
from tweet_classes import ThreadTree
from graph_utils import *
from nlp_utils import *
from feature_utils import *
from classification_utils import classify_test_samples_task_a, classify_test_samples_task_b
from file_utils import *

twitter_test_set_folder = "../rumoureval-2019-test-data/twitter-en-test-data/"


def get_threads_from_story(story):
    folders = glob.glob(twitter_test_set_folder + story + '/*')
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
    thread = ThreadTree(id_thread, source_tweet, reply_tweets, 'data/test_tweets/')
    thread.build_tree(reply_structure)
    return thread


def create_submission_file(task_a_result, task_b_result):
    submission_data = {"subtaskaenglish": task_a_result, "subtaskbenglish": task_b_result}
    submission_data_json = json.dumps(submission_data)
    submission_file = open('./submissions/answer.json', 'w')
    submission_file.write(submission_data_json)
    submission_file.close()


if __name__ == '__main__':
    twitter_stories_folders = glob.glob(twitter_test_set_folder + "*")
    twitter_stories = [
        story_folder.split("\\")[1] for story_folder in twitter_stories_folders
    ]
    print('Stories in test set: ' + str(twitter_stories))

    print('Processing tweet threads...')
    graph_features = {}
    for twitter_story in twitter_stories:
        thread_folders, thread_ids = get_threads_from_story(twitter_story)
        print('Number of threads about ' + twitter_story + ': ' + str(len(thread_ids)))
        for thread_folder, thread_id in zip(thread_folders, thread_ids):
            thread_tree = read_tweet_json(thread_folder, thread_id)
            thread_nx_graph = thread_tree.get_nx_graph()
            graph_features.update(get_graph_features(thread_nx_graph, thread_tree))

    tweet_files = glob.glob('data/test_tweets/*')
    if not isfile('data/datasets/test_dataset_dictionary'):
        tf_idf_document_sets = [{} for i in range(8)]

        language_style_features_tweets = {}
        language_style_features_user_descriptions = {}
        sentiment_features = {}
        extra_features = {}

        for i, tweet_file in enumerate(tweet_files):
            print('Processing tweet #' + str(i + 1) + '...')
            tweet = load_object(tweet_file)
            tweet_id = tweet.id

            tweet_words = tweet.get_words_from_tweet_text()
            tweet_ngrams = get_ngrams_from_words(tweet_words)
            for n, ngrams in enumerate(tweet_ngrams):
                tf_idf_document_sets[n][tweet_id] = ngrams
            tweet_pos_tags = get_pos_tags_for_words(tweet_words)
            tf_idf_document_sets[3][tweet_id] = tweet_pos_tags

            user_description_words = tweet.get_words_from_user_description()
            user_description_ngrams = get_ngrams_from_words(user_description_words)
            for n, ngrams in enumerate(user_description_ngrams):
                tf_idf_document_sets[4 + n][tweet_id] = ngrams
            user_description_pos_tags = get_pos_tags_for_words(user_description_words)
            tf_idf_document_sets[7][tweet_id] = user_description_pos_tags

            tweet_sentences = get_sentences_from_words(tweet_words)
            language_style_features_tweets[tweet_id] = get_extra_language_style_features(tweet_words, tweet_sentences)

            user_description_sentences = get_sentences_from_words(user_description_words)
            language_style_features_user_descriptions[tweet_id] = \
                get_extra_language_style_features(user_description_words, user_description_sentences)

            tweet_words_case_sensitive = tweet.get_words_from_tweet_text(case_sensitive=True)
            tweet_sentences_case_sensitive = get_sentences_from_words(tweet_words_case_sensitive)
            sentiment_features[tweet_id] = get_sentiment_features(tweet_sentences_case_sensitive)
            extra_features[tweet_id] = tweet.get_extra_tweet_features()

        cleaned_vocabularies = load_object('data/cleaned_vocabularies')
        word_cluster_map_tweets = load_object('data/word2vec_documents/word_cluster_map_tweets')
        word_cluster_map_user_descriptions = load_object('data/word2vec_documents/word_cluster_map_user_descriptions')

        tf_idf_feature_sets = []

        for i, (tf_idf_document_set, vocabulary) in enumerate(zip(tf_idf_document_sets, cleaned_vocabularies)):
            if i in (3, 7):
                features = get_pos_features_for_documents(tf_idf_document_set, vocabulary)
            else:
                features = get_tf_idf_features_for_test_documents(tf_idf_document_set, vocabulary)
            tf_idf_feature_sets.append(features)

        word2vec_tweet_features = get_word2vec_word_cluster_features_for_test_documents(word_cluster_map_tweets,
                                                                                        tf_idf_document_sets[0])
        word2vec_user_description_features = get_word2vec_word_cluster_features_for_test_documents(
            word_cluster_map_user_descriptions,
            tf_idf_document_sets[4])

        print('Forming test dataset...')
        dataset = {}
        tweet_ids = list(extra_features.keys())
        for tweet_id in tweet_ids:
            dataset[tweet_id] = language_style_features_tweets[tweet_id]
            dataset[tweet_id].extend(language_style_features_user_descriptions[tweet_id])
            for tf_idf_feature_set in tf_idf_feature_sets:
                dataset[tweet_id].extend(tf_idf_feature_set[tweet_id])
            dataset[tweet_id].extend(word2vec_tweet_features[tweet_id])
            dataset[tweet_id].extend(word2vec_user_description_features[tweet_id])
            dataset[tweet_id].extend(sentiment_features[tweet_id])
            dataset[tweet_id].extend(graph_features[tweet_id])
            dataset[tweet_id].extend(extra_features[tweet_id])
        save_object(dataset, 'data/datasets/test_dataset_dictionary')
        del language_style_features_tweets, language_style_features_user_descriptions, tf_idf_feature_sets, \
            word2vec_tweet_features, word2vec_user_description_features, sentiment_features, \
            graph_features, extra_features
    else:
        dataset = load_object('data/datasets/test_dataset_dictionary')

    feature_labels = load_object('data/features/labels/feature_labels')
    task_a_tweet_ids = []
    task_b_tweet_ids = []
    for tweet_file in tweet_files:
        tweet = load_object(tweet_file)
        tweet_id = tweet.id
        if tweet.source_tweet is not None:
            task_a_tweet_ids.append(tweet_id)
        else:
            task_b_tweet_ids.append(tweet_id)

    if not isfile('data/datasets/test_dataset_task_a.tsv') or not isfile('data/datasets/test_dataset_task_b.tsv'):
        print('Forming test datasets for both tasks...')
        dataset_task_a = []
        feature_labels_task_a = list(feature_labels)
        feature_labels_task_a.extend([label + " DIFFERENCE" for label in feature_labels])
        dataset_task_b = []
        feature_labels_task_b = feature_labels
        for tweet_file in tweet_files:
            tweet = load_object(tweet_file)
            tweet_id = tweet.id
            if tweet.source_tweet is not None:
                source_tweet_id = tweet.source_tweet.id
                task_a_features_for_tweet = dataset[tweet_id]
                task_a_features_for_tweet.extend(
                    difference_of_feature_vectors(dataset[tweet_id], dataset[source_tweet_id]))
                dataset_task_a.append(task_a_features_for_tweet)
            else:
                dataset_task_b.append(dataset[tweet_id])
        del dataset
        dataset_task_a = pd.DataFrame(dataset_task_a, columns=feature_labels_task_a)
        dataset_task_b = pd.DataFrame(dataset_task_b, columns=feature_labels_task_b)
        feature_selector = FeatureSelector(dataset_task_a, None)
        feature_selector.preprocess_dataset()
        dataset_task_a = feature_selector.dataset
        feature_selector = FeatureSelector(dataset_task_b, None)
        feature_selector.preprocess_dataset()
        dataset_task_b = feature_selector.dataset
        dataset_task_a.to_csv('data/datasets/test_dataset_task_a.tsv', sep='\t', header=True, index=False,
                              encoding='utf-8')
        dataset_task_b.to_csv('data/datasets/test_dataset_task_b.tsv', sep='\t', header=True, index=False,
                              encoding='utf-8')
    else:
        del dataset
        dataset_task_a = pd.read_csv('data/datasets/test_dataset_task_a.tsv', sep='\t', index_col=False, header=0,
                                     encoding='utf-8')
        dataset_task_b = pd.read_csv('data/datasets/test_dataset_task_b.tsv', sep='\t', index_col=False, header=0,
                                     encoding='utf-8')

    task_a_comment_feature_labels = load_object('data/features/labels/task_a_comment_feature_labels')
    task_a_query_feature_labels = load_object('data/features/labels/task_a_query_feature_labels')
    task_a_support_deny_feature_labels = load_object('data/features/labels/task_a_support_deny_feature_labels')
    task_b_feature_labels = load_object('data/features/labels/task_b_feature_labels')

    task_a_comment_model = load_object('models/task_a_comment_model')
    task_a_query_model = load_object('models/task_a_query_model')
    task_a_support_deny_model = load_object('models/task_a_support_deny_model')
    task_b_veracity_model = load_object('models/task_b_veracity_model')

    print('Classifying samples for Task A...')
    task_a_class_labels = classify_test_samples_task_a(task_a_tweet_ids, dataset_task_a,
                                                       (task_a_comment_model,
                                                        task_a_query_model,
                                                        task_a_support_deny_model),
                                                       (task_a_comment_feature_labels,
                                                        task_a_query_feature_labels,
                                                        task_a_support_deny_feature_labels))

    print('Classifying samples for Task B...')
    task_b_class_labels = classify_test_samples_task_b(task_b_tweet_ids, dataset_task_b, task_b_veracity_model,
                                                       task_b_feature_labels)

    print('Creating submission file...')
    create_submission_file(task_a_class_labels, task_b_class_labels)
