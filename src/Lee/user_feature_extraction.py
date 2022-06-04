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
        "screen_name_length": uf.get_screen_name_length(user),
        "description_length": uf.get_description_length(user),
        "account_longvity": uf.get_longevity(user),
        "following_count": uf.get_following(user),
        "followers_count": uf.get_followers(user),
        "following_to_followers": uf.get_following_to_followers(user),
        "tweets_count": uf.get_tweets(user),
        "tweets_count_per_day": uf.get_tweets_per_day(user),
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

    the_header = ["id", "screen_name_length", "description_length",
                  "account_longvity", "following_count", "followers_count",
                  "following_to_followers", "tweets_count", "tweets_count_per_day"]
    
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


