import user_features as uf
import csv
import ijson
import pandas as pd
import time
import sys
import os


def get_user_vector(user):
    user_vector = {
        "id": uf.get_user_id(user),
        "profile_description": uf.has_description(user),
        "profile_location": uf.has_location(user),
        "profile_url": uf.has_url(user),
        "verified": uf.is_verified(user),
        "bot_word_in_name": uf.has_bot_word_in_username(user),
        "bot_word_in_screen_name": uf.has_bot_word_in_screen_name(user),
        "bot_word_in_description": uf.has_bot_word_in_description(user),
        "username_length":uf.get_username_length(user),
        "screen_name_length": uf.get_screen_name_length(user),
        "description_length": uf.get_description_length(user),
        "followees_count": uf.get_followees(user),
        "followers_count": uf.get_followers(user),
        "followers_to_followees": uf.get_followers_followees(user),
        "tweets_count": uf.get_tweets(user),
        "listed_count": uf.get_lists(user),
        "numerics_in_username_count": uf.get_number_count_in_username(user),
        "numerics_in_screen_name_count": uf.get_number_count_in_screen_name(user),
        "hashtags_in_username": uf.hashtags_count_in_username(user),
        "hashtags_in_description": uf.hashtags_count_in_description(user),
        "urls_in_description": uf.urls_count_in_description(user),
        "def_image": uf.def_image(user),
        "def_profile": uf.def_profile(user)
    }

    return user_vector


def to_csv(header, dic, filename):
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerows([dic])


def main(argv):
    if argv[1] == '--datasets':
        try:
            name = argv[2]
            return name
        except:
            return "Wrong command!"
    else:
        return "Wrong command!"


if __name__ == '__main__':
    start = time.time()
    dataset_name = main(sys.argv)
    if not os.path.exists('./{}'.format(dataset_name)):
        os.mkdir('./{}'.format(dataset_name))
    filepath = '{}/user_feature.csv'.format(dataset_name)
    
    the_header = ["id","profile_description", "profile_location", "profile_url", "verified",
                  "bot_word_in_name", "bot_word_in_screen_name", "bot_word_in_description",
                  "username_length", "screen_name_length", "description_length", "followees_count",
                  "followers_count", "followers_to_followees", "tweets_count", "listed_count",
                  "numerics_in_username_count", "numerics_in_screen_name_count", "hashtags_in_username",
                  "hashtags_in_description", "urls_in_description", "def_image", "def_profile"]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(the_header)

    if dataset_name == 'Twibot-22':
        json_path = './datasets/Twibot-22/user.json'
    else:
        json_path = './datasets/{}/node.json'.format(dataset_name)

    with open(json_path) as f:
        obj = ijson.items(f, 'item')
        while True:
            try:
                user = obj.__next__()
                if user['id'][0] == 'u':
                    user_feature = get_user_vector(user)
                    header = list(user_feature.keys())
                    to_csv(header, user_feature, filepath)
                else:
                    break
            except StopIteration as e:
                break

    print(time.time()-start)
    
    user_f = pd.read_csv(filepath)
    if os.path.exists("{}/user_to_post.csv".format(dataset_name)):
        up = pd.read_csv("{}/user_to_post.csv".format(dataset_name))
        user_final = list(up['id'])  # user list with label
        user_f = user_f[user_f['id'].isin(user_final)]  # Filter users with label
    else:
        labels = pd.read_csv("./datasets/{}/label.csv".format(dataset_name))
        user_final = list(labels['id'])  # user list with label
        user_f = user_f[user_f['id'].isin(user_final)]  # Filter users with label

    user_f.sort_values(by="id", axis=0, ascending=True, inplace=True, ignore_index=True)
    user_f.to_csv(filepath, index=False)
    print(time.time()-start)


