from dependencies import nx


def get_graph_features(graph, thread):
    thread_tweets = [thread.root.id, ]
    thread_tweets.extend(list(thread.reply_tweets.keys()))
    features = {}
    num_tweets = nx.number_of_nodes(graph)
    dfs_traversal_order = list(nx.dfs_postorder_nodes(graph, thread.root.id))
    dfs_feature = {}
    bfs_successors = list(nx.bfs_successors(graph, thread.root.id))
    bfs_traversal_order = [thread.root.id, ]
    for _, successors in bfs_successors:
        bfs_traversal_order.extend(successors)
    bfs_feature = {}
    for node in graph.nodes():
        dfs_feature[node] = 1 - dfs_traversal_order.index(node) / (num_tweets - 1)
        bfs_feature[node] = 1 - bfs_traversal_order.index(node) / (num_tweets - 1)
    betweenness_centralities = dict(nx.betweenness_centrality(graph))
    closeness_centralities = dict(nx.closeness_centrality(graph))
    degree_centralities = dict(nx.degree_centrality(graph))
    hub_scores, _ = nx.hits(graph, max_iter=1000)
    hub_scores = dict(hub_scores)
    pageranks = dict(nx.pagerank(graph))
    for tweet_id in thread_tweets:
        tweet = thread.find_tweet(tweet_id)
        if tweet is not None and tweet.is_usable():
            features[tweet_id] = [dfs_feature[tweet_id], bfs_feature[tweet_id], degree_centralities[tweet_id],
                                  betweenness_centralities[tweet_id], closeness_centralities[tweet_id],
                                  hub_scores[tweet_id], pageranks[tweet_id]]
    return features
