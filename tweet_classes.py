from dependencies import isfile, html, re, nx
from emoji_utils import *
from date_utils import *
from file_utils import *


class Tweet:
    certainty_map = {
        'certain': 1,
        'somewhat-certain': 0.5,
        'uncertain': 0,
        'underspecified': 0.5,
        None: 0.5,
    }

    evidentiality_map = {
        'no-evidence': 0,
        'reasoning': 1,
        'unverifiable-source-quoted': 2,
        'picture-attached': 3,
        'source-quoted': 4,
        'url-given': 5,
        'witnessed': 6,
        None: 1,
    }

    def __init__(self, tweet_id, tweet, parent_tweet, source_tweet, annotation):
        self.id = tweet_id
        self.tweet = tweet
        self.parent = parent_tweet
        self.children = []
        self.source_tweet = source_tweet
        self.annotation = annotation

    def is_usable(self):
        return self.tweet is not None and 'description' in self.tweet['user'].keys() \
               and self.tweet['user']['description'] is not None

    def get_words_from_tweet_text(self, case_sensitive=False):
        text = html.unescape(self.tweet['text'])
        if not case_sensitive:
            text = text.lower()
        urls = self.tweet['entities']['urls']
        for url in urls:
            url_string = url['url']
            if not case_sensitive:
                url_string = url_string.lower()
            text = re.sub(url_string, ' URL ', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' URL ', text)
        if self.source_tweet is not None:
            source_tweet = self.source_tweet.tweet
            source_tweet_text = source_tweet['text']
            if not case_sensitive:
                source_tweet_text = source_tweet_text.lower()
            text = re.sub(
                r'(“@\w+: .*”)|([RrMm][Tt]\s?@\w+:?\s?' + re.escape(source_tweet_text) + ')', 'QUOTE',
                text)
        text = re.sub(r'(\w+[.|\w])+@(\w+[.])*\w+', ' EMAIL ', text)
        text = re.sub(r'([RrMm][Tt]\s?)?@\w+', ' USER ', text)
        text = re.sub(r'\d+(\.\d+)?', ' NUM ', text)
        text = re.sub('#', ' HASHTAG ', text)
        emojis = get_emoji_from_text(text)
        emojis_concat = ''.join(emojis)
        for emoji in emojis:
            text = re.sub(emoji, ' ' + emoji + ' ', text)
        text = re.sub(r'[^\w\s.,?!:\-+\'' + emojis_concat + ']', ' ', text)
        text = re.sub(r'(.{1,5})\1+', r'\1\1', text)
        text = re.sub(r'\.', ' . ', text)
        text = re.sub(',', ' , ', text)
        text = re.sub(r'\?', ' ? ', text)
        text = re.sub('!', ' ! ', text)
        text = re.sub(':', ' : ', text)
        text = re.sub('-', ' - ', text)
        text = re.sub(r'\+', ' + ', text)
        text = re.sub(r'\'', ' \'', text)
        text = re.sub('\n', ' \n ', text)
        text = re.sub('\r', ' \r ', text)
        words = text.split(" ")
        words = [word for word in words if word != '']
        return words

    def get_words_from_user_description(self, case_sensitive=False):
        user = self.tweet['user']
        text = html.unescape(user['description'])
        if not case_sensitive:
            text = text.lower()
        if 'entities' in user.keys():
            urls = user['entities']['description']['urls']
            for url in urls:
                url_string = url['url']
                if not case_sensitive:
                    url_string = url_string.lower()
                text = re.sub(url_string, ' URL ', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' URL ', text)
        text = re.sub(r'(\w+[.|\w])+@(\w+[.])*\w+', ' EMAIL ', text)
        text = re.sub(r'([RrMm][Tt]\s?)@\w+', ' USER ', text)
        text = re.sub(r'\d+(\.\d+)?', ' NUM ', text)
        text = re.sub('#', ' HASHTAG ', text)
        emojis = get_emoji_from_text(text)
        emojis_concat = ''.join(emojis)
        for emoji in emojis:
            text = re.sub(emoji, ' ' + emoji + ' ', text)
        text = re.sub(r'[^\w\s.,?!:\-+\'' + emojis_concat + ']', ' ', text)
        text = re.sub(r'(.{1,5})\1+', r'\1\1', text)
        text = re.sub(r'\.', ' . ', text)
        text = re.sub(',', ' , ', text)
        text = re.sub(r'\?', ' ? ', text)
        text = re.sub('!', ' ! ', text)
        text = re.sub(':', ' : ', text)
        text = re.sub('-', ' - ', text)
        text = re.sub(r'\+', ' + ', text)
        text = re.sub(r'\'', ' \'', text)
        text = re.sub('\n', ' \n ', text)
        text = re.sub('\r', ' \r ', text)
        words = text.split(" ")
        words = [word for word in words if word != '']
        return words

    def get_extra_tweet_features(self):
        tweet_data = self.tweet
        user = tweet_data['user']
        features = []
        if self.annotation is not None and 'evidentiality' in self.annotation.keys():
            features.append(Tweet.evidentiality_map[self.annotation['evidentiality']])
        else:
            features.append(Tweet.evidentiality_map[None])
        if self.annotation is not None and 'certainty' in self.annotation.keys():
            features.append(Tweet.certainty_map[self.annotation['certainty']])
        else:
            features.append(Tweet.certainty_map[None])
        features.append(len(tweet_data['text']))
        features.append(tweet_data['favorite_count'])
        features.append(tweet_data['retweet_count'])
        features.append(get_age_of_date(twitter_utc_string_to_datetime(tweet_data['created_at'])))
        features.append(int(user['verified']))
        features.append(user['followers_count'])
        features.append(user['statuses_count'])
        features.append(user['friends_count'])
        features.append(user['favourites_count'])
        features.append(user['listed_count'])
        features.append(get_age_of_date(twitter_utc_string_to_datetime(user['created_at'])))
        features.append(int(user['profile_use_background_image']))
        features.append(user['default_profile_image'])
        features.append(int(user['profile_text_color'], 16))
        features.append(int(user['profile_sidebar_fill_color'], 16))
        features.append(int(user['profile_sidebar_border_color'], 16))
        features.append(int(user['profile_background_color'], 16))
        features.append(int(user['profile_link_color'], 16))
        return features

    def __str__(self):
        return self.tweet['user']['screen_name'] + ": " + self.tweet['text'] if self.tweet is not None else "TWEET " \
                                                                                                            "MISSING "


class ThreadTree:
    tweet_annotations = None

    def __init__(self, source_tweet_id, source_tweet, reply_tweets, tweet_folder):
        self.source_tweet = source_tweet
        self.reply_tweets = reply_tweets
        if ThreadTree.tweet_annotations is not None and source_tweet_id in ThreadTree.tweet_annotations.keys():
            self.root = Tweet(source_tweet_id, source_tweet, None, None,
                              ThreadTree.tweet_annotations[source_tweet_id])
        else:
            self.root = Tweet(source_tweet_id, source_tweet, None, None, None)
        self.tweet_folder = tweet_folder

    @staticmethod
    def load_annotations_file():
        ThreadTree.tweet_annotations = load_object('data/annotations')

    def __build_tree_r(self, node, reply_structure):
        if len(reply_structure) == 0:
            return
        for child_tweet_id, child_reply_structure in reply_structure.items():
            if child_tweet_id in self.reply_tweets.keys():
                child_tweet = self.reply_tweets[child_tweet_id]
            else:
                child_tweet = None
            if ThreadTree.tweet_annotations is not None and child_tweet_id in ThreadTree.tweet_annotations.keys():
                child = Tweet(child_tweet_id, child_tweet, node, self.root,
                              ThreadTree.tweet_annotations[child_tweet_id])
            else:
                child = Tweet(child_tweet_id, child_tweet, node, self.root, None)
            node.children.append(child)
            if not isfile(self.tweet_folder + child_tweet_id):
                save_object(child, self.tweet_folder + child_tweet_id)
            self.__build_tree_r(child, child_reply_structure)

    def build_tree(self, reply_structure):
        source_tweet_id, source_reply_structure = tuple(
            reply_structure.items())[0]
        if not isfile(self.tweet_folder + source_tweet_id):
            save_object(self.root, self.tweet_folder + source_tweet_id)
        self.__build_tree_r(self.root, source_reply_structure)

    def __find_tweet_r(self, node, id_tweet):
        if node.id == id_tweet:
            return node
        for child in node.children:
            search_result = self.__find_tweet_r(child, id_tweet)
            if search_result is not None:
                return search_result
        return None

    def find_tweet(self, id_tweet):
        return self.__find_tweet_r(self.root, id_tweet)

    def __build_nx_graph_r(self, node, graph):
        graph.add_node(node.id)
        for child in node.children:
            graph.add_edge(node.id, child.id)
            graph = self.__build_nx_graph_r(child, graph)

        return graph

    def get_nx_graph(self):
        return self.__build_nx_graph_r(self.root, nx.Graph())

    def __print_tree_r(self, node, depth):
        print("--" * depth + "> " + str(node))
        for child in node.children:
            self.__print_tree_r(child, depth + 1)

    def print_tree(self):
        self.__print_tree_r(self.root, 0)
