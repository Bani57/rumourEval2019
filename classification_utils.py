from dependencies import cosine, Counter, seed, sample, train_test_split, GaussianNB, KNeighborsClassifier, \
    LogisticRegression, SVC, MLPClassifier, RandomForestClassifier, VotingClassifier, accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score, log_loss, GridSearchCV

seed(57)

num_cpu_cores = 6

classifiers = {'Naive Bayes': GaussianNB(),
               'KNN': KNeighborsClassifier(
                   n_neighbors=30,
                   weights="distance",
                   algorithm="auto",
                   metric='euclidean',
                   n_jobs=-1),
               'Logistic Regression': LogisticRegression(
                   C=1e-2,
                   solver='liblinear',
                   tol=1e-8,
                   max_iter=1000,
                   class_weight="balanced",
                   random_state=57,
                   n_jobs=1),
               'SVM': SVC(
                   C=1e-2,
                   kernel='rbf',
                   gamma="scale",
                   decision_function_shape="ovr",
                   class_weight="balanced",
                   random_state=57),
               'Neural Network': MLPClassifier(
                   hidden_layer_sizes=(200, 100),
                   activation="tanh",
                   solver="sgd",
                   learning_rate="invscaling",
                   learning_rate_init=0.01,
                   max_iter=1000,
                   early_stopping=True,
                   validation_fraction=0.1,
                   random_state=57),
               'Random Forest': RandomForestClassifier(
                   n_estimators=1000,
                   criterion="entropy",
                   min_samples_leaf=5,
                   class_weight="balanced",
                   random_state=57,
                   n_jobs=-1)}

param_grids = {
    'Naive Bayes': {'var_smoothing': [10 ** value for value in range(-18, 0)]},
    'KNN': {'n_neighbors': [30, ], 'algorithm': ['brute', ], 'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', cosine], 'n_jobs': [-1, ]},
    'Logistic Regression': {'C': [10 ** value for value in range(0, -11, -1)],
                            'solver': ['liblinear', ], 'max_iter': [1e4, ],
                            'tol': [1e-8, ], 'class_weight': ['balanced', ], 'n_jobs': [1, ]},
    'SVM': [{'C': [10 ** value for value in range(0, -11, -1)], 'kernel': ['poly', ], 'degree': range(1, 6),
             'class_weight': ['balanced', ]}, {'C': [10 ** value for value in range(0, -11, -1)], 'kernel': ['rbf', ],
                                               'gamma': ['scale', ], 'class_weight': ['balanced', ]}],
    'Neural Network': [{'hidden_layer_sizes': [(value,) for value in (10, 25, 50, 100, 250)],
                        'activation': ['tanh', ], 'solver': ['adam'],
                        'alpha': [10 ** value for value in range(0, -12, -2)],
                        'learning_rate': ['constant', ],
                        'learning_rate_init': [1e-2, ],
                        'max_iter': [1000, ],
                        'tol': [1e-8, ]},
                       {'hidden_layer_sizes': [(value,) for value in (10, 25, 50, 100, 250)],
                        'activation': ['tanh', ], 'solver': ['sgd'],
                        'alpha': [10 ** value for value in range(0, -12, -2)],
                        'learning_rate': ['constant', 'invscaling', 'adaptive'],
                        'learning_rate_init': [1e-2, ],
                        'max_iter': [1000, ],
                        'tol': [1e-8, ]}
                       ],
    'Random Forest': {'n_estimators': (10, 25, 50, 100, 250, 500, 1000), 'criterion': ['gini', 'entropy'],
                      'max_depth': [None, 10, 25, 50, 100], 'min_samples_split': (2, 3, 5, 10),
                      'min_samples_leaf': (2, 3, 5, 10), 'n_jobs': [-1, ]}
}


def get_class_distribution(class_labels):
    return dict(Counter(class_labels))


def split_dataset_task_a_one_vs_rest(dataset, labels):
    comment_labels = ['not-comment' if label != 'comment' else label for label in labels]
    tuple_1 = (dataset, comment_labels)
    query_labels = []
    query_dataset_indexes = []
    for i, label in enumerate(comment_labels):
        if label == 'not-comment':
            query_labels.append('not-query' if labels[i] != 'query' else labels[i])
            query_dataset_indexes.append(i)
    query_dataset = dataset.iloc[query_dataset_indexes, :]
    tuple_2 = (query_dataset, query_labels)
    support_deny_labels = []
    support_deny_dataset_indexes = []
    for i, label in zip(query_dataset_indexes, query_labels):
        if label == 'not-query':
            support_deny_labels.append(labels[i])
            support_deny_dataset_indexes.append(i)
    support_deny_dataset = dataset.iloc[support_deny_dataset_indexes, :]
    tuple_3 = (support_deny_dataset, support_deny_labels)
    return tuple_1, tuple_2, tuple_3


task_a_classes_to_binary_map = {
    'comment': 1,
    'not-comment': 0,
    'query': 1,
    'not-query': 0,
    'support': 1,
    'deny': 0
}


def get_task_a_binary_labels(labels):
    return [task_a_classes_to_binary_map[label] for label in labels]


def get_indexes_of_samples_with_label(labels):
    num_rows = len(labels)
    indexes = {}
    for i in range(num_rows):
        class_of_sample = labels[i]
        indexes.setdefault(class_of_sample, [])
        indexes[class_of_sample].append(i)
    return indexes


def get_balanced_datasets(dataset, labels, num_datasets=10):
    class_distribution = get_class_distribution(labels)
    min_class, min_class_samples = min(class_distribution.items(), key=lambda x: x[1])
    indexes = get_indexes_of_samples_with_label(labels)
    min_class_indexes = indexes[min_class]
    balanced_dataset_tuples = []
    for i in range(num_datasets):
        balanced_dataset = dataset.iloc[min_class_indexes, :]
        balanced_labels = [min_class, ] * min_class_samples
        for label in set(labels):
            if label != min_class:
                indexes_sample = sample(indexes[label], min_class_samples)
                balanced_dataset = balanced_dataset.append(dataset.iloc[indexes_sample, :], sort=False)
                balanced_labels.extend([label, ] * min_class_samples)
        balanced_dataset_tuples.append((balanced_dataset, balanced_labels))
    return balanced_dataset_tuples


def get_dataset_tuples_for_validation(dataset, labels, n=10):
    dataset_tuples = []
    for i in range(n):
        dataset_tuples.append(tuple(train_test_split(dataset, labels, shuffle=True, stratify=labels, test_size=0.3)))
    return dataset_tuples


def score_classifier(classifier, dataset_tuples, probability=False):
    scores = [0, ] * 5
    if probability:
        scores.append(0)
    for train_set, validation_set, train_labels, validation_labels in dataset_tuples:
        classifier.fit(train_set, train_labels)
        predicted_labels = classifier.predict(validation_set)
        scores[0] += accuracy_score(validation_labels, predicted_labels)
        scores[1] += precision_score(validation_labels, predicted_labels, average='weighted')
        scores[2] += recall_score(validation_labels, predicted_labels, average='weighted')
        scores[3] += f1_score(validation_labels, predicted_labels, average='weighted')
        scores[4] += roc_auc_score(validation_labels, predicted_labels, average='weighted')
        if probability:
            predicted_label_probabilities = classifier.predict_proba(validation_set)
            scores[5] += log_loss(validation_labels, predicted_label_probabilities)
    scores = [score / len(dataset_tuples) for score in scores]
    return scores


def train_model(dataset, labels, balance_dataset=True, optimize_parameters=False, classify_with_probabilities=False):
    if balance_dataset:
        print('Finding best balanced dataset...')
        balanced_dataset_tuples = get_balanced_datasets(dataset, labels, 30)
        best_dataset_tuple = None
        best_dataset_tuple_score = 0
        for balanced_dataset, balanced_labels in balanced_dataset_tuples:
            dataset_tuples = get_dataset_tuples_for_validation(balanced_dataset, balanced_labels)
            naive_bayes = GaussianNB()
            scores = score_classifier(naive_bayes, dataset_tuples, classify_with_probabilities)
            relevant_score = scores[5] if classify_with_probabilities else scores[3]
            if relevant_score > best_dataset_tuple_score:
                best_dataset_tuple_score = relevant_score
                best_dataset_tuple = (balanced_dataset, balanced_labels)

        print('Score of best dataset: ' + str(best_dataset_tuple_score))
        best_dataset, best_labels = best_dataset_tuple
        del balanced_dataset_tuples, best_dataset_tuple

    else:
        best_dataset, best_labels = dataset, labels

    dataset_tuples = get_dataset_tuples_for_validation(best_dataset, best_labels)

    scores = {}
    if classify_with_probabilities:
        scoring = 'neg_log_loss'
        ensemble_model_voting = 'soft'
        classifiers['SVM'].set_params(probability=True)
    else:
        scoring = 'f1_weighted'
        ensemble_model_voting = 'hard'

    if optimize_parameters:
        optimized_classifiers = {}
        for classifier_name, classifier_model in classifiers.items():
            print('Optimizing and training ' + classifier_name + '...')
            classifier_grid_search = GridSearchCV(classifier_model, param_grids[classifier_name],
                                                  scoring=scoring, cv=3, refit=True, verbose=1)
            classifier_grid_search.fit(best_dataset, best_labels)
            print('Best found parameter combination: ' + str(classifier_grid_search.best_params_))
            best_classifier = classifier_grid_search.best_estimator_
            optimized_classifiers[classifier_name] = best_classifier
            print('Scoring ' + classifier_name + '...')
            scores[classifier_name] = score_classifier(best_classifier, dataset_tuples, classify_with_probabilities)
        sorted_classifiers = sorted(scores.items(), key=lambda x: x[1][3])
        sorted_classifiers = [classifier_name for classifier_name, _ in sorted_classifiers]
        ensemble_weights = [sorted_classifiers.index(classifier_name) + 1 for classifier_name in
                            optimized_classifiers.keys()]
        ensemble_model = VotingClassifier(estimators=list(optimized_classifiers.items()), voting=ensemble_model_voting,
                                          weights=ensemble_weights)
    else:
        for classifier_name, classifier_model in classifiers.items():
            print('Training and scoring ' + classifier_name + '...')
            scores[classifier_name] = score_classifier(classifier_model, dataset_tuples, classify_with_probabilities)
        sorted_classifiers = sorted(scores.items(), key=lambda x: x[1][3])
        sorted_classifiers = [classifier_name for classifier_name, _ in sorted_classifiers]
        ensemble_weights = [sorted_classifiers.index(classifier_name) + 1 for classifier_name in classifiers.keys()]
        ensemble_model = VotingClassifier(estimators=list(classifiers.items()), voting=ensemble_model_voting,
                                          weights=ensemble_weights)

    print('Training and scoring Ensemble model...')
    scores['Ensemble model'] = score_classifier(ensemble_model, dataset_tuples, classify_with_probabilities)
    ensemble_model.fit(best_dataset, best_labels)
    return ensemble_model, scores


def classify_test_samples_task_a(tweet_ids, dataset, models, lists_of_relevant_features):
    comment_dataset = dataset.loc[:, lists_of_relevant_features[0]]
    comment_model = models[0]
    predicted_labels = comment_model.predict(comment_dataset)
    predicted_labels = ['comment' if label == 1 else 'not-comment' for label in predicted_labels]
    not_comment_indexes = []
    for i, label in enumerate(predicted_labels):
        if label == 'not-comment':
            not_comment_indexes.append(i)
    if len(not_comment_indexes) > 0:
        query_dataset = dataset.loc[not_comment_indexes, lists_of_relevant_features[1]]
        query_model = models[1]
        query_predicted_labels = query_model.predict(query_dataset)
        for i, label in zip(not_comment_indexes, query_predicted_labels):
            predicted_labels[i] = 'query' if label == 1 else 'not-query'
        not_query_indexes = []
        for i, label in enumerate(predicted_labels):
            if label == 'not-query':
                not_query_indexes.append(i)
        if len(not_query_indexes) > 0:
            support_deny_dataset = dataset.loc[not_query_indexes, lists_of_relevant_features[2]]
            support_deny_model = models[2]
            support_deny_predicted_labels = support_deny_model.predict(support_deny_dataset)
            for i, label in zip(not_query_indexes, support_deny_predicted_labels):
                predicted_labels[i] = 'support' if label == 1 else 'deny'
    return {tweet_id: label for tweet_id, label in zip(tweet_ids, predicted_labels)}


def classify_test_samples_task_b(tweet_ids, dataset, model, relevant_features):
    veracity_dataset = dataset.loc[:, relevant_features]
    predicted_labels = model.predict(veracity_dataset)
    predicted_label_probabilities = model.predict_proba(veracity_dataset)
    result = {}
    for tweet_id, label, label_probabilities in zip(tweet_ids, predicted_labels, predicted_label_probabilities):
        higher_probability = max(label_probabilities)
        result[tweet_id] = ['true' if label == 1 else 'false', higher_probability]
    return result
