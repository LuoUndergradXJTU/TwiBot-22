import re
import math
import collections
import emoji
import langid
import string
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from tqdm import tqdm
from datetime import datetime
from zhon.hanzi import punctuation


MAX_NEIGHBOR = 250
MONTH_DICT = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}


def str_entropy(inp_str):
    counter_char = collections.Counter(inp_str)
    entropy = 0
    for c, ctn in counter_char.items():
        _p = float(ctn) / len(inp_str)
        entropy += -1 * _p * math.log(_p, 2)
    return entropy


def feature_preprocess(node, edge, dataset):
    X = []
    id_map = dict()
    num_user = 0
    ind_first = True
    for i in range(len(node)):
        id_map[node[i]['id']] = i
        if node[i]['id'][0] == 't' and ind_first:
            num_user = i
            ind_first = False
    if num_user == 0:
        num_user = len(node)
    follow = edge.values[edge['relation'].values == 'follow']
    graph = nx.Graph()
    print("Building graph...", flush=True)
    graph.add_nodes_from(list(range(num_user)))
    for i in tqdm(range(follow.shape[0])):
        if follow[i, 0] in id_map and follow[i, 2] in id_map:
            graph.add_edge(id_map[follow[i, 0]], id_map[follow[i, 2]])

    ego_networks = []
    print("Building ego networks...", flush=True)
    for i in tqdm(range(num_user)):
        ego_node = id_map[node[i]['id']]
        G = nx.Graph()
        G.add_node(ego_node)
        for ind1, adj1 in enumerate(list(graph.adj[ego_node])):
            if ind1 >= MAX_NEIGHBOR:
                break
            for ind2, adj2 in enumerate(list(graph.adj[adj1])):
                if ind2 >= MAX_NEIGHBOR:
                    break
                G.add_edge(adj1, adj2)
            G.add_edge(ego_node, adj1)
        ego_networks.append(G)

    post = edge.values[edge['relation'].values == 'post']
    tweet_map = dict()
    for i in range(num_user):
        tweet_map[i] = []
    for i in range(post.shape[0]):
        tweet_map[id_map[post[i, 0]]].append(str(node[id_map[post[i, 2]]]['text']))

    # screen name length
    screen_name_length = []
    lack_index_1 = []
    print("Building feature 1/50...", flush=True)
    for i in tqdm(range(num_user)):
        if node[i]['username'] is not None:
            screen_name_length.append(len(node[i]['username']))
        else:
            lack_index_1.append(i)
            screen_name_length.append(0)
    mean_value = int(sum(screen_name_length) / (len(screen_name_length) - len(lack_index_1)))
    for i in lack_index_1:
        screen_name_length[i] = mean_value
    X.append(screen_name_length)

    # default profile image
    default_profile_image = []
    print("Building feature 2/50...", flush=True)
    for i in tqdm(range(num_user)):
        if node[i]['profile_image_url'] is not None:
            default_profile_image.append(1)
        else:
            default_profile_image.append(0)
    X.append(default_profile_image)

    # entropy screen name
    entropy_screen_name = []
    lack_index_3 = []
    print("Building feature 3/50...", flush=True)
    for i in tqdm(range(num_user)):
        if node[i]['username'] is not None:
            entropy_screen_name.append(str_entropy(node[i]['username']))
        else:
            lack_index_3.append(i)
            entropy_screen_name.append(0.0)
    mean_value = sum(entropy_screen_name) / float(len(entropy_screen_name) - len(lack_index_3))
    for i in lack_index_3:
        entropy_screen_name[i] = mean_value
    X.append(entropy_screen_name)

    # has location
    has_location = []
    print("Building feature 4/50...", flush=True)
    for i in tqdm(range(num_user)):
        if node[i]['location'] is not None:
            has_location.append(1)
        else:
            has_location.append(0)
    X.append(has_location)

    # total tweets
    total_tweets = []
    lack_index_5 = []
    print("Building feature 5/50...", flush=True)
    for i in tqdm(range(num_user)):
        if node[i]['public_metrics']['tweet_count'] is not None:
            total_tweets.append(node[i]['public_metrics']['tweet_count'])
        else:
            lack_index_5.append(i)
            total_tweets.append(0)
    mean_value = int(sum(total_tweets) / (len(total_tweets) - len(lack_index_5)))
    for i in lack_index_3:
        total_tweets[i] = mean_value
    X.append(total_tweets)

    # % w/ default image
    w_default_image = []
    print("Building feature 6/50...", flush=True)
    for i in tqdm(range(num_user)):
        image_count = 0
        for j in list(ego_networks[i].nodes):
            if node[j]['profile_image_url'] is not None:
                image_count += 1.0
        w_default_image.append(image_count / len(list(ego_networks[i].nodes)))
    X.append(w_default_image)

    # median tweets
    median_tweets = []
    lack_index_7 = []
    print("Building feature 7/50...", flush=True)
    for i in tqdm(range(num_user)):
        tweet_count_list = []
        for j in list(ego_networks[i].nodes):
            if node[j]['public_metrics']['tweet_count'] is not None:
                tweet_count_list.append(node[j]['public_metrics']['tweet_count'])
        if len(tweet_count_list) == 0:
            median_tweets.append(0)
            lack_index_7.append(i)
        else:
            tweet_count_list.sort()
            median_tweets.append(
                (tweet_count_list[len(tweet_count_list) // 2] + tweet_count_list[~(len(tweet_count_list) // 2)]) // 2)
    mean_value = int(sum(median_tweets) / (len(median_tweets) - len(lack_index_7)))
    for i in lack_index_7:
        median_tweets[i] = mean_value
    X.append(median_tweets)

    # mean age (day)
    mean_age = []
    lack_index_8 = []
    print("Building feature 8/50...", flush=True)
    for i in tqdm(range(num_user)):
        age_list = []
        for j in list(ego_networks[i].nodes):
            if node[j]['created_at'] is not None:
                time_list = node[j]['created_at'].split(' ')
                # time_list.pop()
                time_split = []
                for k in time_list:
                    if len(k) > 0:
                        time_split.append(k)
                if len(time_list) == 1:
                    age_list.append(0.0)
                    continue
                born_date = datetime(int(time_split[-1]), MONTH_DICT[time_split[1]], int(time_split[2]),
                                     int(time_split[3].split(':')[0]), int(time_split[3].split(':')[1]),
                                     int(time_split[3].split(':')[2]))
                now_date = datetime.now()
                age_list.append((now_date - born_date).days)
        if len(age_list) == 0:
            mean_age.append(0.0)
            lack_index_8.append(i)
        else:
            mean_age.append(sum(age_list) / len(age_list))
    mean_value = sum(mean_age) / (len(mean_age) - len(lack_index_8))
    for i in lack_index_8:
        mean_age[i] = mean_value
    X.append(mean_age)

    # % w/ description
    w_description = []
    print("Building feature 9/50...", flush=True)
    for i in tqdm(range(num_user)):
        description_count = 0
        for j in list(ego_networks[i].nodes):
            if node[j]['description'] is not None:
                description_count += 1.0
        w_description.append(description_count / len(list(ego_networks[i].nodes)))
    X.append(w_description)

    # number of friends
    number_of_friends = []
    lack_index_10 = []
    print("Building feature 10/50...", flush=True)
    for i in tqdm(range(num_user)):
        if node[i]['public_metrics']['following_count'] is not None:
            number_of_friends.append(node[i]['public_metrics']['following_count'])
        else:
            number_of_friends.append(0)
            lack_index_10.append(i)
    mean_value = int(sum(number_of_friends) / (len(number_of_friends) - len(lack_index_10)))
    for i in lack_index_10:
        number_of_friends[i] = mean_value
    X.append(number_of_friends)

    # number of followers
    number_of_followers = []
    lack_index_11 = []
    print("Building feature 11/50...", flush=True)
    for i in tqdm(range(num_user)):
        if node[i]['public_metrics']['followers_count'] is not None:
            number_of_followers.append(node[i]['public_metrics']['followers_count'])
        else:
            number_of_followers.append(0)
            lack_index_11.append(i)
    mean_value = int(sum(number_of_followers) / (len(number_of_followers) - len(lack_index_11)))
    for i in lack_index_11:
        number_of_followers[i] = mean_value
    X.append(number_of_followers)

    # number nodes of E
    E_nodes_num = []
    print("Building feature 12/50...", flush=True)
    for i in tqdm(range(num_user)):
        E_nodes_num.append(len(list(ego_networks[i].nodes)))
    X.append(E_nodes_num)

    # number edges of E
    E_edges_num = []
    print("Building feature 13/50...", flush=True)
    for i in tqdm(range(num_user)):
        E_edges_num.append(len(list(ego_networks[i].edges)))
    X.append(E_edges_num)

    # density of E
    E_density = []
    print("Building feature 14/50...", flush=True)
    for i in tqdm(range(num_user)):
        E_density.append(nx.density(ego_networks[i]))
    X.append(E_density)

    # components of E
    E_components = []
    print("Building feature 15/50...", flush=True)
    for i in tqdm(range(num_user)):
        E_components.append(nx.number_connected_components(ego_networks[i]))
    X.append(E_components)

    # largest components of E
    E_largest_components = []
    print("Building feature 16/50...", flush=True)
    for i in tqdm(range(num_user)):
        E_largest_components.append(max([len(j) for j in nx.connected_components(ego_networks[i])]))
    X.append(E_largest_components)

    # degree centrality of E
    E_degree_centrality = []
    print("Building feature 17/50...", flush=True)
    for i in tqdm(range(num_user)):
        cent = nx.degree_centrality(ego_networks[i])
        if len(cent) == 0:
            E_degree_centrality.append(0.0)
        else:
            E_degree_centrality.append(sum(cent.values()) / len(cent))
    X.append(E_degree_centrality)

    # number of isolates in E
    E_isolates_num = []
    print("Building feature 18/50...", flush=True)
    for i in tqdm(range(num_user)):
        E_isolates_num.append(nx.number_of_isolates(ego_networks[i]))
    X.append(E_isolates_num)

    # number of dyad isolates in E
    E_dyad_isolates_num = []
    print("Building feature 19/50...", flush=True)
    for i in tqdm(range(num_user)):
        dyad_isolates_count = 0
        for j in nx.connected_components(ego_networks[i]):
            if len(j) == 2:
                dyad_isolates_count += 1
        E_dyad_isolates_num.append(dyad_isolates_count)
    X.append(E_dyad_isolates_num)

    # number of triad isolates in E
    E_triad_isolates_num = []
    print("Building feature 20/50...", flush=True)
    for i in tqdm(range(num_user)):
        triad_isolates_count = 0
        for j in nx.connected_components(ego_networks[i]):
            if len(j) == 3:
                triad_isolates_count += 1
        E_triad_isolates_num.append(triad_isolates_count)
    X.append(E_triad_isolates_num)

    # number of >4 isolates in E
    E_four_isolates_num = []
    print("Building feature 21/50...", flush=True)
    for i in tqdm(range(num_user)):
        four_isolates_count = 0
        for j in nx.connected_components(ego_networks[i]):
            if len(j) >= 4:
                four_isolates_count += 1
        E_four_isolates_num.append(four_isolates_count)
    X.append(E_four_isolates_num)

    # clustering coefficient of E
    E_clustering_coefficient = []
    print("Building feature 22/50...", flush=True)
    for i in tqdm(range(num_user)):
        E_clustering_coefficient.append(nx.average_clustering(ego_networks[i]))
    X.append(E_clustering_coefficient)

    # transitivity of E
    E_transitivity = []
    print("Building feature 23/50...", flush=True)
    for i in tqdm(range(num_user)):
        E_transitivity.append(nx.transitivity(ego_networks[i]))
    X.append(E_transitivity)

    # K-betweenness centrality of E
    E_betweenness_centrality = []
    print("Building feature 24/50...", flush=True)
    for i in tqdm(range(num_user)):
        betweenness_centrality = nx.betweenness_centrality(ego_networks[i],
                                                           k=min(500, ego_networks[i].number_of_nodes()))
        E_betweenness_centrality.append(sum([betweenness_centrality[j] for j in betweenness_centrality.keys()]) / len(
            betweenness_centrality.keys()))
    X.append(E_betweenness_centrality)

    # # eigenvector centrality of E
    # E_eigenvector_centrality = []
    # print("Building feature 25/49...", flush=True)
    # for i in tqdm(range(num_user)):
    #     eigenvector_centrality = nx.eigenvector_centrality(ego_networks[i])
    #     E_eigenvector_centrality.append(sum([eigenvector_centrality[j] for j in eigenvector_centrality.keys()]) / len(
    #         eigenvector_centrality.keys()))
    # X.append(E_eigenvector_centrality)

    # louvain group of E
    E_louvain_num = []
    E_largest_louvain = []
    print("Building feature 26,27/50...", flush=True)
    for i in tqdm(range(num_user)):
        louvain_group = community_louvain.best_partition(ego_networks[i])
        louvain_list = np.array([louvain_group[j] for j in louvain_group.keys()])
        louvain_num = np.max(louvain_list) + 1
        E_louvain_num.append(louvain_num)
        E_largest_louvain.append(max([(louvain_list == j).sum() for j in range(louvain_num)]))
    X.append(E_louvain_num)
    X.append(E_largest_louvain)

    # ego effective size of E
    E_ego_effective_size = []
    print("Building feature 28/50...", flush=True)
    for i in tqdm(range(num_user)):
        E_ego_effective_size.append(nx.effective_size(ego_networks[i])[i])
    X.append(E_ego_effective_size)

    # median followers of E
    E_median_followers = []
    print("Building feature 29/50...", flush=True)
    lack_index_26 = []
    for i in tqdm(range(num_user)):
        median_followers = []
        for j in ego_networks[i].nodes():
            if node[j]['public_metrics']['followers_count'] is not None:
                median_followers.append(node[j]['public_metrics']['followers_count'])
        if len(median_followers) == 0:
            E_median_followers.append(0)
            lack_index_26.append(i)
        else:
            E_median_followers.append(
                (median_followers[len(median_followers) // 2] + median_followers[~(len(median_followers) // 2)]) / 2.0)
    mean_value = sum(E_median_followers) / (len(E_median_followers) - len(lack_index_26))
    for i in lack_index_26:
        E_median_followers[i] = mean_value
    X.append(E_median_followers)

    # median friends of E
    E_median_friends = []
    lack_index_27 = []
    print("Building feature 30/50...", flush=True)
    for i in tqdm(range(num_user)):
        median_friends = []
        for j in ego_networks[i].nodes():
            if node[j]['public_metrics']['following_count'] is not None:
                median_friends.append(node[j]['public_metrics']['following_count'])
        if len(median_friends) == 0:
            E_median_friends.append(0)
            lack_index_27.append(i)
        else:
            E_median_friends.append(
                (median_friends[len(median_friends) // 2] + median_friends[~(len(median_friends) // 2)]) / 2.0)
    mean_value = sum(E_median_friends) / (len(E_median_friends) - len(lack_index_27))
    for i in lack_index_27:
        E_median_friends[i] = mean_value
    X.append(E_median_friends)

    # is last status retweet?
    last_retweet_or_not = []
    print("Building feature 31/50...", flush=True)
    for i in tqdm(range(num_user)):
        if len(tweet_map[i]) > 0:
            if tweet_map[i][-1][0:4] == 'RT @':
                last_retweet_or_not.append(1)
            else:
                last_retweet_or_not.append(0)
        else:
            last_retweet_or_not.append(0)
    X.append(last_retweet_or_not)

    # # same language?
    # same_language_or_not = []
    # print("Building feature 32/50...", flush=True)
    # for i in tqdm(range(num_user)):
    #     if len(tweet_map[i]) > 0:
    #         tweet_list = []
    #         for j in range(len(tweet_map[i])):
    #             tweet_list.append(tweet_map[i][j])
    #         tweet_object = ''.join(tweet_list)
    #     else:
    #         tweet_object = ''
    #     user_list = []
    #     if node[i]['name'] is not None:
    #         user_list.append(node[i]['name'])
    #     if node[i]['username'] is not None:
    #         user_list.append(node[i]['username'])
    #     if node[i]['description'] is not None:
    #         user_list.append(node[i]['description'])
    #     if len(user_list) == 0:
    #         user_object = ''
    #     else:
    #         user_object = ''.join(user_list)
    #     if len(tweet_object) == 0 or len(user_object) == 0:
    #         same_language_or_not.append(1)
    #     else:
    #         if langid.classify(user_object)[0] == langid.classify(tweet_object)[0]:
    #             same_language_or_not.append(1)
    #         else:
    #             same_language_or_not.append(0)
    # X.append(same_language_or_not)

    # hashtags last status
    hashtags_last_status = []
    print("Building feature 33/50...", flush=True)
    for i in tqdm(range(num_user)):
        if len(tweet_map[i]) > 0:
            hashtags_last_status.append(len(re.findall('#', tweet_map[i][-1])))
        else:
            hashtags_last_status.append(0)
    X.append(hashtags_last_status)

    # mentions last status
    mentions_last_status = []
    print("Building feature 34/50...", flush=True)
    for i in tqdm(range(num_user)):
        if len(tweet_map[i]) > 0:
            mentions_last_status.append(len(re.findall('@', tweet_map[i][-1])))
        else:
            mentions_last_status.append(0)
    X.append(mentions_last_status)

    # bot reference
    bot_reference = []
    print("Building feature 35/50...", flush=True)
    for i in tqdm(range(num_user)):
        if node[i]['description'] is not None:
            if re.match('[Bb][Oo][Tt]', str(node[i]['description'])) is not None:
                bot_reference.append(1)
                continue
        if node[i]['name'] is not None:
            if re.match('[Bb][Oo][Tt]', str(node[i]['name'])) is not None:
                bot_reference.append(1)
                continue
        if node[i]['username'] is not None:
            if re.match('[Bb][Oo][Tt]', str(node[i]['username'])) is not None:
                bot_reference.append(1)
                continue
        bot_reference.append(0)
    X.append(bot_reference)

    # mean mentions
    mean_mentions = []
    print("Building feature 36/50...", flush=True)
    for i in tqdm(range(num_user)):
        if len(tweet_map[i]) > 0:
            mentions_count = 0
            for j in range(len(tweet_map[i])):
                mentions_count += len(re.findall('@', tweet_map[i][j]))
            mean_mentions.append(mentions_count / len(tweet_map[i]))
        else:
            mean_mentions.append(0.0)
    X.append(mean_mentions)

    # mean hashtags
    mean_hashtags = []
    print("Building feature 37/50...", flush=True)
    for i in tqdm(range(num_user)):
        if len(tweet_map[i]) > 0:
            hashtags_count = 0
            for j in range(len(tweet_map[i])):
                hashtags_count += len(re.findall('#', tweet_map[i][j]))
            mean_hashtags.append(hashtags_count / len(tweet_map[i]))
        else:
            mean_hashtags.append(0.0)
    X.append(mean_hashtags)

    # max mentions
    max_mentions = []
    print("Building feature 38/50...", flush=True)
    for i in tqdm(range(num_user)):
        max_mention = 0
        for j in range(len(tweet_map[i])):
            max_mention = max(max_mention, len(re.findall('@', tweet_map[i][j])))
        max_mentions.append(max_mention)
    X.append(max_mentions)

    # max hashtags
    max_hashtags = []
    print("Building feature 39/50...", flush=True)
    for i in tqdm(range(num_user)):
        max_hashtag = 0
        for j in range(len(tweet_map[i])):
            max_hashtag = max(max_hashtag, len(re.findall('#', tweet_map[i][j])))
        max_hashtags.append(max_hashtag)
    X.append(max_hashtags)

    # # number of languages
    # languages_num = []
    # print("Building feature 40/50...", flush=True)
    # for i in tqdm(range(num_user)):
    #     languages_dict = dict()
    #     if len(tweet_map[i]) > 0:
    #         for j in range(len(tweet_map[i])):
    #             languages_dict[langid.classify(tweet_map[i][j])[0]] = 0
    #     if node[i]['name'] is not None:
    #         languages_dict[langid.classify(node[i]['name'])[0]] = 0
    #     if node[i]['username'] is not None:
    #         languages_dict[langid.classify(node[i]['username'])[0]] = 0
    #     if node[i]['description'] is not None:
    #         languages_dict[langid.classify(node[i]['description'])[0]] = 0
    #     languages_num.append(len(languages_dict))
    # X.append(languages_num)

    # fraction retweets
    fraction_retweets = []
    print("Building feature 41/50...", flush=True)
    for i in tqdm(range(num_user)):
        if len(tweet_map[i]) > 0:
            retweet_count = 0
            for j in range(len(tweet_map[i])):
                if tweet_map[i][j][0:4] == 'RT @':
                    retweet_count += 1
            fraction_retweets.append(retweet_count / len(tweet_map[i]))
        else:
            fraction_retweets.append(0.0)
    X.append(fraction_retweets)

    # # number of languages in E
    # E_languages_num = []
    # print("Building feature 42/50...", flush=True)
    # for i in tqdm(range(num_user)):
    #     languages_sum = 0
    #     for j in ego_networks[i].nodes():
    #         languages_dict = dict()
    #         if len(tweet_map[j]) > 0:
    #             for k in range(len(tweet_map[j])):
    #                 languages_dict[langid.classify(tweet_map[j][k])[0]] = 0
    #         if node[j]['name'] is not None:
    #             languages_dict[langid.classify(node[j]['name'])[0]] = 0
    #         if node[j]['username'] is not None:
    #             languages_dict[langid.classify(node[j]['username'])[0]] = 0
    #         if node[j]['description'] is not None:
    #             languages_dict[langid.classify(node[j]['description'])[0]] = 0
    #         languages_sum += len(languages_dict)
    #     E_languages_num.append(languages_sum)
    # X.append(E_languages_num)

    # mean emoji per tweet
    mean_emoji_per_tweet = []
    print("Building feature 43/50...", flush=True)
    for i in tqdm(range(num_user)):
        emoji_count = 0
        for tweet in tweet_map[i]:
            for char in tweet:
                if char in emoji.UNICODE_EMOJI['en']:
                    emoji_count += 1
        if len(tweet_map[i]) == 0:
            mean_emoji_per_tweet.append(0)
        else:
            mean_emoji_per_tweet.append(emoji_count / len(tweet_map[i]))
    X.append(mean_emoji_per_tweet)

    # mean mentions in E
    E_mean_mentions = []
    print("Building feature 44/50...", flush=True)
    for i in tqdm(range(num_user)):
        mentions_sum = 0
        tweets_sum = 0
        for j in ego_networks[i].nodes():
            if len(tweet_map[j]) > 0:
                for k in range(len(tweet_map[j])):
                    mentions_sum += len(re.findall('@', tweet_map[j][k]))
                tweets_sum += len(tweet_map[j])
        if tweets_sum == 0:
            E_mean_mentions.append(0.0)
        else:
            E_mean_mentions.append(mentions_sum / tweets_sum)
    X.append(E_mean_mentions)

    # mean hashtags in E
    E_mean_hashtags = []
    print("Building feature 45/50...", flush=True)
    for i in tqdm(range(num_user)):
        hashtags_sum = 0
        tweets_sum = 0
        for j in ego_networks[i].nodes():
            if len(tweet_map[j]) > 0:
                for k in range(len(tweet_map[j])):
                    hashtags_sum += len(re.findall('#', tweet_map[j][k]))
                tweets_sum += len(tweet_map[j])
        if tweets_sum == 0:
            E_mean_hashtags.append(0.0)
        else:
            E_mean_hashtags.append(hashtags_sum / tweets_sum)
    X.append(E_mean_hashtags)

    # % retweets in E
    E_fraction_retweets = []
    print("Building feature 46/50...", flush=True)
    for i in tqdm(range(num_user)):
        retweets_sum = 0
        tweets_sum = 0
        for j in ego_networks[i].nodes():
            if len(tweet_map[j]) > 0:
                for k in range(len(tweet_map[j])):
                    if tweet_map[j][k][0:4] == 'RT @':
                        retweets_sum += 1
                tweets_sum += len(tweet_map[j])
        if tweets_sum == 0:
            E_fraction_retweets.append(0.0)
        else:
            E_fraction_retweets.append(retweets_sum / tweets_sum)
    X.append(E_fraction_retweets)

    # account age (day)
    account_age = []
    lack_index_28 = []
    print("Building feature 47/50...", flush=True)
    for i in tqdm(range(num_user)):
        if node[i]['created_at'] is not None:
            time_list = node[i]['created_at'].split(' ')
            # time_list.pop()
            time_split = []
            for j in time_list:
                if len(j) > 0:
                    time_split.append(j)
            if len(time_list) == 1:
                account_age.append(0.0)
                lack_index_28.append(i)
                continue
            born_date = datetime(int(time_split[-1]), MONTH_DICT[time_split[1]], int(time_split[2]),
                                 int(time_split[3].split(':')[0]), int(time_split[3].split(':')[1]),
                                 int(time_split[3].split(':')[2]))
            now_date = datetime.now()
            account_age.append((now_date - born_date).days)
        else:
            account_age.append(0.0)
            lack_index_28.append(i)
    mean_value = sum(account_age) / (len(account_age) - len(lack_index_28))
    for i in lack_index_28:
        account_age[i] = mean_value
    X.append(account_age)

    # mean jaccard/cosine similarity
    jaccard_similarity = []
    cosine_similarity = []
    lack_jaccard = []
    lack_cosine = []
    print("Building feature 48,49/50...", flush=True)
    for i in tqdm(range(num_user)):
        vocab = set()
        word_map = dict()
        for j in ego_networks[i].nodes():
            word_map[j] = dict()
            tweet_content = ''
            for k in range(len(tweet_map[j])):
                tweet_content = tweet_content + tweet_map[j][k]
            tweet_content = tweet_content.lower()
            tweet_content = re.sub(r'\n', '', tweet_content)
            url_list = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', tweet_content)
            word_map[j][0] = len(url_list)
            for k in url_list:
                tweet_content = re.sub(k, '', tweet_content)
            for char in string.punctuation:
                tweet_content = tweet_content.replace(char, '')
            for char in punctuation:
                tweet_content = tweet_content.replace(char, '')
            word_list = tweet_content.split(' ')
            for word in word_list:
                if len(word) != 0:
                    vocab.add(word)
                    if word not in word_map[j].keys():
                        word_map[j][word] = 1.0
                    else:
                        word_map[j][word] += 1.0
        jaccard_similarity_sum = []
        cosine_similarity_sum = []
        for e in ego_networks[i].edges():
            set_0 = set()
            set_1 = set()
            for j in word_map[e[0]].keys():
                set_0.add(j)
            for j in word_map[e[1]].keys():
                set_1.add(j)
            if len(set_0 | set_1) == 0:
                jaccard_similarity_sum.append(0.0)
            else:
                jaccard_similarity_sum.append(len(set_0 & set_1) / len(set_0 | set_1))
            word_vec_0 = [word_map[e[0]][0]]
            word_vec_1 = [word_map[e[1]][0]]
            for j in vocab:
                if j in word_map[e[0]].keys():
                    word_vec_0.append(word_map[e[0]][j])
                else:
                    word_vec_0.append(0.0)
                if j in word_map[e[1]].keys():
                    word_vec_1.append(word_map[e[1]][j])
                else:
                    word_vec_1.append(0.0)
            word_vec_0 = np.array(word_vec_0, dtype=np.float)
            word_vec_1 = np.array(word_vec_1, dtype=np.float)
            if np.linalg.norm(word_vec_0) * np.linalg.norm(word_vec_1) == 0:
                cosine_similarity_sum.append(0.0)
            else:
                cosine_similarity_sum.append(
                    np.sum(word_vec_0 * word_vec_1) / np.linalg.norm(word_vec_0) / np.linalg.norm(word_vec_1))
        if len(jaccard_similarity_sum) == 0:
            lack_jaccard.append(i)
            jaccard_similarity.append(0.0)
        else:
            jaccard_similarity.append(sum(jaccard_similarity_sum) / len(jaccard_similarity_sum))
        if len(cosine_similarity_sum) == 0:
            lack_cosine.append(i)
            cosine_similarity.append(0.0)
        else:
            cosine_similarity.append(sum(cosine_similarity_sum) / len(cosine_similarity_sum))
    mean_jaccard = sum(jaccard_similarity) / (len(jaccard_similarity) - len(lack_jaccard))
    mean_cosine = sum(cosine_similarity) / (len(cosine_similarity) - len(lack_cosine))
    for i in lack_jaccard:
        jaccard_similarity[i] = mean_jaccard
    for i in lack_cosine:
        cosine_similarity[i] = mean_cosine
    X.append(jaccard_similarity)
    X.append(cosine_similarity)

    # % many likes and few followers
    many_likes_few_followers = []
    print("Building feature 50/50...", flush=True)
    for i in tqdm(range(num_user)):
        user_count = 0
        for j in ego_networks[i].nodes():
            retweet_count = 0
            follower_count = 0
            following_count = 0
            for k in range(len(tweet_map[j])):
                if tweet_map[j][k][0:4] == 'RT @':
                    retweet_count += 1
            if node[j]['public_metrics']['followers_count'] is not None:
                follower_count = node[j]['public_metrics']['followers_count']
            if node[j]['public_metrics']['following_count'] is not None:
                following_count = node[j]['public_metrics']['following_count']
            if retweet_count > 2 * max(follower_count, following_count):
                user_count += 1
        if ego_networks[i].number_of_nodes() == 0:
            many_likes_few_followers.append(0.0)
        else:
            many_likes_few_followers.append(user_count / ego_networks[i].number_of_nodes())
    X.append(many_likes_few_followers)

    Feature = np.array(X).T
    Feature_matrix = pd.DataFrame(Feature)
    Feature_matrix.to_csv('feature_matrix_' + dataset + '.csv')

    return Feature
