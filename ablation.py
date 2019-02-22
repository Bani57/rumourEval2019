from dependencies import np, pd, isfile
from file_utils import *
from classification_utils import *

ablation_labels = ["Without language style features", "Without TF-IDF and Word2Vec features",
                   "Without sentiment features", "Without graph features", "Without extra features"]

evaluation_metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']

language_style_feature_labels = ["NUMBER OF WORDS", "NUMBER OF SENTENCES", "AVERAGE WORDS PER SENTENCE",
                                 "LANGUAGE PERCENTAGE", "MEAN WORD LENGTH", "WORD LENGTH VARIANCE",
                                 "DICTIONARY PERCENTAGE", "UNIQUE WORDS PERCENTAGE"]
sentiment_feature_labels = ["MEAN POSITIVE SENTIMENT", "MEAN NEUTRAL SENTIMENT", "MEAN NEGATIVE SENTIMENT",
                            "POSITIVE SENTIMENT VARIANCE", "NEUTRAL SENTIMENT VARIANCE",
                            "NEGATIVE SENTIMENT VARIANCE"]
graph_feature_labels = ["DFS PRIORITY", "BFS PRIORITY", "DEGREE CENTRALITY", "BETWEENNESS CENTRALITY",
                        "CLOSENESS CENTRALITY",
                        "HUB SCORE", "PAGERANK"]
extra_feature_labels = ['EVIDENTIALITY', 'CERTAINTY', 'CHARACTER COUNT', 'FAVORITE COUNT', 'RETWEET COUNT',
                        'AGE', 'USER IS VERIFIED', 'USER FOLLOWERS COUNT', 'USER STATUSES COUNT',
                        'USER FRIENDS COUNT', 'USER FAVORITES COUNT', 'USER LISTED COUNT', 'USER PROFILE AGE',
                        'USER USES A BACKGROUD IMAGE', 'USER HAS THE DEFAULT PROFILE IMAGE',
                        'USER PROFILE TEXT COLOR', 'USER PROFILE SIDEBAR FILL COLOR',
                        'USER PROFILE SIDEBAR BORDER COLOR', 'USER PROFILE BACKGROUND COLOR',
                        'USER PROFILE LINK COLOR']


def get_ablation_datasets(dataset):
    datasets = []
    language_style_columns = [column for column in dataset.columns if
                              any(label in column for label in language_style_feature_labels)]
    tf_idf_word2vec_columns = [column for column in dataset.columns if "TF-IDF" in column or "WORD2VEC" in column]
    sentiment_columns = [column for column in dataset.columns if
                         any(label in column for label in sentiment_feature_labels)]
    graph_columns = [column for column in dataset.columns if
                     any(label in column for label in graph_feature_labels)]
    extra_columns = [column for column in dataset.columns if
                     any(label in column for label in extra_feature_labels)]
    dataset_without_columns = dataset.drop(language_style_columns, axis=1)
    datasets.append(dataset_without_columns)
    dataset_without_columns = dataset.drop(tf_idf_word2vec_columns, axis=1)
    datasets.append(dataset_without_columns)
    dataset_without_columns = dataset.drop(sentiment_columns, axis=1)
    datasets.append(dataset_without_columns)
    dataset_without_columns = dataset.drop(graph_columns, axis=1)
    datasets.append(dataset_without_columns)
    dataset_without_columns = dataset.drop(extra_columns, axis=1)
    datasets.append(dataset_without_columns)
    return datasets


if __name__ == "__main__":
    if not isfile('scores/ablation/task_a_comment_scores.tsv'):
        comment_dataset = pd.read_csv('data/datasets/task_a_comment_dataset.tsv', sep='\t', index_col=False, header=0,
                                      encoding='utf-8')
        comment_class_labels = load_object('data/class_labels/task_a_comment_class_labels')

        ablation_datasets = get_ablation_datasets(comment_dataset)
        del comment_dataset

        comment_ablation_scores = {}
        for ablation_label, ablation_dataset in zip(ablation_labels, ablation_datasets):
            _, scores = train_model(ablation_dataset, comment_class_labels, optimize_parameters=False)
            comment_ablation_scores[ablation_label] = scores['Ensemble model']
        comment_ablation_scores = pd.DataFrame(np.array(list(comment_ablation_scores.values()), dtype='float'),
                                               index=ablation_labels,
                                               columns=evaluation_metrics)
        comment_ablation_scores.to_csv('scores/ablation/task_a_comment_scores.tsv', sep='\t', index=True, header=True,
                                       encoding='utf-8')

    if not isfile('scores/ablation/task_a_query_scores.tsv'):
        query_dataset = pd.read_csv('data/datasets/task_a_query_dataset.tsv', sep='\t', index_col=False, header=0,
                                    encoding='utf-8')
        query_class_labels = load_object('data/class_labels/task_a_query_class_labels')

        ablation_datasets = get_ablation_datasets(query_dataset)
        del query_dataset

        query_ablation_scores = {}
        for ablation_label, ablation_dataset in zip(ablation_labels, ablation_datasets):
            _, scores = train_model(ablation_dataset, query_class_labels, optimize_parameters=False)
            query_ablation_scores[ablation_label] = scores['Ensemble model']
        query_ablation_scores = pd.DataFrame(np.array(list(query_ablation_scores.values()), dtype='float'),
                                             index=ablation_labels,
                                             columns=evaluation_metrics)
        query_ablation_scores.to_csv('scores/ablation/task_a_query_scores.tsv', sep='\t', index=True, header=True,
                                     encoding='utf-8')

    if not isfile('scores/ablation/task_a_support_deny_scores.tsv'):
        support_deny_dataset = pd.read_csv('data/datasets/task_a_support_deny_dataset.tsv', sep='\t', index_col=False,
                                           header=0,
                                           encoding='utf-8')
        support_deny_class_labels = load_object('data/class_labels/task_a_support_deny_class_labels')

        ablation_datasets = get_ablation_datasets(support_deny_dataset)
        del support_deny_dataset

        support_deny_ablation_scores = {}
        for ablation_label, ablation_dataset in zip(ablation_labels, ablation_datasets):
            _, scores = train_model(ablation_dataset, support_deny_class_labels, optimize_parameters=False)
            support_deny_ablation_scores[ablation_label] = scores['Ensemble model']
        support_deny_ablation_scores = pd.DataFrame(
            np.array(list(support_deny_ablation_scores.values()), dtype='float'),
            index=ablation_labels,
            columns=evaluation_metrics)
        support_deny_ablation_scores.to_csv('scores/ablation/task_a_support_deny_scores.tsv', sep='\t', index=True,
                                            header=True,
                                            encoding='utf-8')

    evaluation_metrics.append('Cross-entropy loss')
    if not isfile('scores/ablation/task_b_scores.tsv'):
        veracity_dataset = pd.read_csv('data/datasets/dataset_task_b.tsv', sep='\t', index_col=False,
                                       header=0,
                                       encoding='utf-8')
        veracity_class_labels = load_object('data/class_labels/class_labels_task_b')

        ablation_datasets = get_ablation_datasets(veracity_dataset)
        del veracity_dataset

        veracity_ablation_scores = {}
        for ablation_label, ablation_dataset in zip(ablation_labels, ablation_datasets):
            _, scores = train_model(ablation_dataset, veracity_class_labels, optimize_parameters=False,
                                    balance_dataset=False, classify_with_probabilities=True)
            veracity_ablation_scores[ablation_label] = scores['Ensemble model']
        veracity_ablation_scores = pd.DataFrame(np.array(list(veracity_ablation_scores.values()), dtype='float'),
                                                index=ablation_labels,
                                                columns=evaluation_metrics)
        veracity_ablation_scores.to_csv('scores/ablation/task_b_scores.tsv', sep='\t', index=True,
                                        header=True,
                                        encoding='utf-8')
