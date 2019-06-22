from dependencies import np, pd, plt, train_test_split, sample, accuracy_score, roc_auc_score, roc_curve
from file_utils import load_object
from nlp_utils import get_words_in_word2vec_cluster


def plot_roc_curve(model, dataset, labels, model_name, plot_destination_file):
    predicted_labels = model.predict(dataset)
    predicted_label_probabilities = model.predict_proba(dataset)
    accuracy = round(accuracy_score(labels, predicted_labels), 4)
    roc_score = round(roc_auc_score(labels, predicted_label_probabilities[:, 1], average='weighted'), 4)
    fpr, tpr, _ = roc_curve(labels, predicted_label_probabilities[:, 1], pos_label=1)
    plt.figure(figsize=(16, 9), dpi=1280 // 16)
    plt.plot(fpr, tpr, color='blue', lw=3)
    plt.plot([0, 1], [0, 1], lw=3, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - ' + str(model_name) +
              '\nAccuracy: ' + str(accuracy) +
              '\nROC AUC score: ' + str(roc_score),
              fontsize=24)
    plt.savefig(plot_destination_file, dpi='figure', format='png')


def analyze_class_distribution_for_features(dataset, features, labels, model_name, plot_destination_file):
    class_differences = []
    for feature in features:
        vector = list(dataset[feature])
        positive_vector = []
        negative_vector = []
        for i, label in enumerate(labels):
            if label == 1:
                positive_vector.append(vector[i])
            else:
                negative_vector.append(vector[i])
        class_differences.append(np.mean(positive_vector) - np.mean(negative_vector))
    class_differences = class_differences / sum(map(abs, class_differences))
    plt.figure(figsize=(16, 9), dpi=1280 // 16)
    plt.title("Class distribution difference for features - " + str(model_name), fontsize=18)
    x = np.arange(len(features))
    plt.bar(x, class_differences, align='center', tick_label=features)
    plt.xticks(x, features, rotation='vertical')
    plt.xlabel("Features")
    plt.ylabel("Positive vs Negative class average feature value difference")
    plt.axhline(y=0, color='black', lw=1)
    plt.tight_layout()
    plt.savefig(plot_destination_file, dpi='figure', format='png')


def get_random_predicted_samples(model, dataset, labels):
    samples_amount = 5
    correct_predictions = []
    incorrect_predictions = []
    num_samples, _ = dataset.shape
    i = 0

    while i < num_samples and \
            (len(correct_predictions) < samples_amount or len(incorrect_predictions) < samples_amount):
        feature_vector = np.array(dataset.iloc[i, :]).reshape(1, -1)
        predicted_label = model.predict(feature_vector)[0]
        correct_label = labels[i]
        prediction = (i, predicted_label, correct_label)
        if predicted_label == correct_label:
            correct_predictions.append(prediction)
        else:
            incorrect_predictions.append(prediction)
        i += 1

    for k, (i, predicted_label, correct_label) in enumerate(correct_predictions):
        tweet_id = dataset.index[i]
        tweet = load_object("data/tweets/" + str(tweet_id))
        tweet_prediction = (tweet, predicted_label, correct_label)
        correct_predictions[k] = tweet_prediction
    for k, (i, predicted_label, correct_label) in enumerate(incorrect_predictions):
        tweet_id = dataset.index[i]
        tweet = load_object("data/tweets/" + str(tweet_id))
        tweet_prediction = (tweet, predicted_label, correct_label)
        incorrect_predictions[k] = tweet_prediction

    if len(correct_predictions) > 5:
        correct_predictions = sample(correct_predictions, samples_amount)
    if len(incorrect_predictions) > 5:
        incorrect_predictions = sample(incorrect_predictions, samples_amount)

    return correct_predictions, incorrect_predictions


def print_tweet_samples(tweet_samples):
    for tweet, predicted_class, correct_class in tweet_samples:
        print("Tweet:", tweet)
        print("Predicted class:", predicted_class)
        print("Tweet class:", correct_class)
        print()


if __name__ == "__main__":
    top_comment_features = load_object("data/features/top/top_comment_features")
    print("Top comment features:", top_comment_features)
    top_query_features = load_object("data/features/top/top_query_features")
    print("Top query features:", top_query_features)
    top_support_deny_features = load_object("data/features/top/top_support_deny_features")
    print("Top support vs deny features:", top_support_deny_features)
    top_veracity_features = load_object("data/features/top/top_veracity_features")
    print("Top veracity features:", top_veracity_features)
    print()

    comment_feature_labels = load_object('data/features/labels/task_a_comment_feature_labels')
    query_feature_labels = load_object('data/features/labels/task_a_query_feature_labels')
    support_deny_feature_labels = load_object('data/features/labels/task_a_support_deny_feature_labels')
    veracity_feature_labels = load_object('data/features/labels/task_b_feature_labels')

    unique_comment_features = sorted(set(comment_feature_labels).difference(query_feature_labels)
                                     .difference(support_deny_feature_labels).difference(veracity_feature_labels))

    unique_query_features = sorted(set(query_feature_labels).difference(comment_feature_labels)
                                   .difference(support_deny_feature_labels).difference(veracity_feature_labels))

    unique_support_deny_features = sorted(set(support_deny_feature_labels).difference(comment_feature_labels)
                                          .difference(query_feature_labels).difference(veracity_feature_labels))

    unique_veracity_features = sorted(set(veracity_feature_labels).difference(comment_feature_labels)
                                      .difference(query_feature_labels).difference(support_deny_feature_labels))

    print("Unique comment features: ", unique_comment_features)
    print("Unique query features: ", unique_query_features)
    print("Unique support vs deny features: ", unique_support_deny_features)
    print("Unique veracity features: ", unique_veracity_features)
    print()

    comment_dataset = pd.read_csv('data/datasets/task_a_comment_dataset.tsv', sep='\t', index_col=0, header=0,
                                  encoding='utf-8')
    comment_labels = load_object('data/class_labels/task_a_comment_class_labels')

    query_dataset = pd.read_csv('data/datasets/task_a_query_dataset.tsv', sep='\t', index_col=0, header=0,
                                encoding='utf-8')
    query_labels = load_object('data/class_labels/task_a_query_class_labels')

    support_deny_dataset = pd.read_csv('data/datasets/task_a_support_deny_dataset.tsv', sep='\t', index_col=0, header=0,
                                       encoding='utf-8')
    support_deny_labels = load_object('data/class_labels/task_a_support_deny_class_labels')

    veracity_dataset = pd.read_csv('data/datasets/dataset_task_b.tsv', sep='\t', index_col=0, header=0,
                                   encoding='utf-8')
    veracity_labels = load_object('data/class_labels/class_labels_task_b')

    word_cluster_map_tweets = load_object('data/word2vec_documents/word_cluster_map_tweets')
    word_cluster_map_user_descriptions = load_object('data/word2vec_documents/word_cluster_map_user_descriptions')

    print("Number of word clusters for tweet text:", max(word_cluster_map_tweets.values()) + 1)
    print("Number of word clusters for user description text:", max(word_cluster_map_user_descriptions.values()) + 1)
    print()

    comment_word_clusters = [feature_label for feature_label in unique_comment_features
                             if "WORD2VEC CLUSTER" in feature_label]
    query_word_clusters = [feature_label for feature_label in unique_query_features
                           if "WORD2VEC CLUSTER" in feature_label]
    support_deny_word_clusters = [feature_label for feature_label in unique_support_deny_features
                                  if "WORD2VEC CLUSTER" in feature_label]
    veracity_word_clusters = [feature_label for feature_label in unique_veracity_features
                              if "WORD2VEC CLUSTER" in feature_label]

    words_in_cluster = get_words_in_word2vec_cluster(word_cluster_map_tweets, 0)

    relevant_comment_features = [". TF-IDF", ". END TF-IDF", "! END TF-IDF", "? TF-IDF",
                                 "QUOTE TF-IDF", "HASHTAG TF-IDF", "URL TF-IDF",
                                 "USER i TF-IDF DIFFERENCE",
                                 "[USER DESCRIPTION] AVERAGE WORDS PER SENTENCE",
                                 "[USER DESCRIPTION] i am TF-IDF DIFFERENCE",
                                 "MEAN WORD LENGTH", "DICTIONARY PERCENTAGE",
                                 "MEAN NEUTRAL SENTIMENT",
                                 "BFS PRIORITY", "HUB SCORE", "PAGERANK",
                                 "AGE", "USER PROFILE AGE", "USER HAS THE DEFAULT PROFILE IMAGE",
                                 "EVIDENTIALITY", "CERTAINTY"]

    relevant_query_features = ["? TF-IDF", "? END TF-IDF",
                               "QUOTE TF-IDF", "HASHTAG TF-IDF", "URL TF-IDF", "USER TF-IDF",
                               "but TF-IDF DIFFERENCE", "said TF-IDF DIFFERENCE", "why TF-IDF",
                               "what TF-IDF", "where TF-IDF", "who TF-IDF",
                               "confirm TF-IDF DIFFERENCE", "need TF-IDF DIFFERENCE",
                               "[USER DESCRIPTION] AVERAGE WORDS PER SENTENCE",
                               "MEAN WORD LENGTH", "DICTIONARY PERCENTAGE",
                               "MEAN NEUTRAL SENTIMENT",
                               "BFS PRIORITY", "HUB SCORE", "PAGERANK",
                               "AGE", "USER PROFILE AGE",
                               "EVIDENTIALITY", "CERTAINTY"]

    relevant_support_deny_features = ["not TF-IDF", "' TF-IDF", "! END TF-IDF", "! TF-IDF",
                                      "QUOTE TF-IDF", "HASHTAG TF-IDF", "URL TF-IDF", "USER TF-IDF",
                                      "you TF-IDF", "said TF-IDF DIFFERENCE", "believe TF-IDF DIFFERENCE",
                                      "know TF-IDF", "oh TF-IDF",
                                      "such TF-IDF", "yeah TF-IDF", "understand TF-IDF DIFFERENCE",
                                      "fuck TF-IDF", "thank TF-IDF", "god TF-IDF",
                                      "[USER DESCRIPTION] AVERAGE WORDS PER SENTENCE",
                                      "[USER DESCRIPTION] love books TF-IDF DIFFERENCE",
                                      "MEAN WORD LENGTH", "DICTIONARY PERCENTAGE",
                                      "MEAN POSITIVE SENTIMENT", "MEAN NEGATIVE SENTIMENT",
                                      "BFS PRIORITY", "HUB SCORE", "PAGERANK",
                                      "AGE", "USER PROFILE AGE",
                                      "EVIDENTIALITY", "CERTAINTY"]

    relevant_veracity_features = ["AVERAGE WORDS PER SENTENCE", "MEAN WORD LENGTH",
                                  "LANGUAGE PERCENTAGE", "DICTIONARY PERCENTAGE", "UNIQUE WORDS PERCENTAGE",
                                  ": TF-IDF", "NUM TF-IDF", "HASHTAG TF-IDF", "URL TF-IDF", "USER TF-IDF",
                                  "[USER DESCRIPTION] bbc TF-IDF",
                                  "[USER DESCRIPTION] news TF-IDF",
                                  "[USER DESCRIPTION] delivering you TF-IDF",
                                  "[USER DESCRIPTION] insightful analysis TF-IDF",
                                  "[USER DESCRIPTION] breaking news TF-IDF",
                                  "[USER DESCRIPTION] facebook TF-IDF",
                                  "[USER DESCRIPTION] EMAIL TF-IDF",
                                  "AGE", "USER PROFILE AGE",
                                  "EVIDENTIALITY", "CERTAINTY"]

    analyze_class_distribution_for_features(comment_dataset, relevant_comment_features, comment_labels,
                                            "Comment model", "plots/comment_features_class_distribution.png")
    analyze_class_distribution_for_features(query_dataset, relevant_query_features, query_labels,
                                            "Query model", "plots/query_features_class_distribution.png")
    analyze_class_distribution_for_features(support_deny_dataset, relevant_support_deny_features, support_deny_labels,
                                            "Support vs Deny model",
                                            "plots/support_deny_features_class_distribution.png")
    analyze_class_distribution_for_features(veracity_dataset, relevant_veracity_features, veracity_labels,
                                            "Veracity model", "plots/veracity_features_class_distribution.png")

    comment_model = load_object('models/task_a_comment_model')
    query_model = load_object('models/task_a_query_model')
    support_deny_model = load_object('models/task_a_support_deny_model')
    veracity_model = load_object('models/task_b_veracity_model')

    comment_train_set, comment_test_set, comment_train_labels, comment_test_labels = \
        train_test_split(comment_dataset, comment_labels, shuffle=True, stratify=comment_labels, test_size=0.3)
    comment_model.fit(comment_train_set, comment_train_labels)
    query_train_set, query_test_set, query_train_labels, query_test_labels = \
        train_test_split(query_dataset, query_labels, shuffle=True, stratify=query_labels, test_size=0.3)
    query_model.fit(query_train_set, query_train_labels)
    support_deny_train_set, support_deny_test_set, support_deny_train_labels, support_deny_test_labels = \
        train_test_split(support_deny_dataset, support_deny_labels,
                         shuffle=True, stratify=support_deny_labels, test_size=0.3)
    support_deny_model.fit(support_deny_train_set, support_deny_train_labels)
    veracity_train_set, veracity_test_set, veracity_train_labels, veracity_test_labels = \
        train_test_split(veracity_dataset, veracity_labels, shuffle=True, stratify=veracity_labels, test_size=0.3)
    veracity_model.fit(veracity_train_set, veracity_train_labels)

    comment_correct_samples, comment_incorrect_samples = \
        get_random_predicted_samples(comment_model, comment_test_set, comment_test_labels)
    query_correct_samples, query_incorrect_samples = \
        get_random_predicted_samples(query_model, query_test_set, query_test_labels)
    support_deny_correct_samples, support_deny_incorrect_samples = \
        get_random_predicted_samples(support_deny_model, support_deny_test_set, support_deny_test_labels)
    veracity_correct_samples, veracity_incorrect_samples = \
        get_random_predicted_samples(veracity_model, veracity_test_set, veracity_test_labels)

    comment_tweet_samples = comment_correct_samples + comment_incorrect_samples
    print("Comment tweet samples:")
    print_tweet_samples(comment_tweet_samples)
    print()

    query_tweet_samples = query_correct_samples + query_incorrect_samples
    print("Query tweet samples:")
    print_tweet_samples(query_tweet_samples)
    print()

    support_deny_tweet_samples = support_deny_correct_samples + support_deny_incorrect_samples
    print("Support vs Deny tweet samples:")
    print_tweet_samples(support_deny_tweet_samples)
    print()

    veracity_tweet_samples = veracity_correct_samples + veracity_incorrect_samples
    print("Veracity tweet samples:")
    print_tweet_samples(veracity_tweet_samples)
    print()
