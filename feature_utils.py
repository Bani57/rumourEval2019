from dependencies import seed, RandomForestClassifier, SelectFromModel, pd

seed(57)

random_forest = RandomForestClassifier(
    n_estimators=1000,
    criterion="entropy",
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=57,
    n_jobs=-1)

reply_annotation_to_task_a_label_map = {
    'agreed': 'support',
    'disagreed': 'deny',
    'appeal-for-more-information': 'query',
    'comment': 'comment'
}


def get_task_a_label(annotation):
    return reply_annotation_to_task_a_label_map[annotation]


def difference_of_feature_vectors(vector_1, vector_2):
    return [feature_1 - feature_2 for feature_1, feature_2 in zip(vector_1, vector_2)]


def remove_missing_values(vector, replacement=-1):
    return [replacement if value is None else value for value in vector]


def categorical_feature_to_numeric(vector):
    unique_values = list(set(vector))
    numeric_values_map = {value: i for i, value in enumerate(unique_values)}
    return [numeric_values_map[value] for value in vector]


def min_max_normalize_feature(vector, new_min=0, new_max=1):
    min_value = min(vector)
    max_value = max(vector)
    if min_value == max_value:
        return [-1, ] * len(vector)
    else:
        return [new_min + (new_max - new_min) * (value - min_value) / (max_value - min_value) for value in vector]


class FeatureSelector:
    def __init__(self, dataset, labels):
        self.dataset = dataset.copy()
        self.labels = labels
        self.feature_names = dataset.columns.values.tolist()
        self.feature_importances = None

    def preprocess_dataset(self):
        for feature_name in self.feature_names:
            feature = self.dataset[feature_name]
            if any(type(value) is str for value in feature):
                feature = categorical_feature_to_numeric(feature)
            if any(value is None for value in feature):
                feature = remove_missing_values(feature)
            feature = min_max_normalize_feature(feature, -1, 1)
            self.dataset[feature_name] = feature

    def remove_uninformative_features(self):
        model = SelectFromModel(random_forest)
        model.fit(self.dataset, self.labels)
        self.feature_importances = model.estimator_.feature_importances_
        features_retained = model.get_support(indices=True)
        features_retained = [self.feature_names[retained_feature] for retained_feature in features_retained]
        self.feature_names = features_retained
        self.dataset = model.transform(self.dataset)
        self.dataset = pd.DataFrame(self.dataset, columns=features_retained)

    def get_top_n_features(self, n=10):
        if self.feature_importances is not None:
            feature_scores = [(feature_name, score) for feature_name, score in
                              zip(self.feature_names, self.feature_importances)]
            feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
            top_features = []
            print('Best features by importance score: ')
            for i in range(n):
                feature_name, feature_score = feature_scores[i]
                print(feature_name + ": " + str(feature_score))
                top_features.append(feature_name)
            return top_features

    def perform_whole_pipeline(self):
        self.preprocess_dataset()
        self.remove_uninformative_features()
        return self.dataset, self.feature_names
