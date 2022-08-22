from MyClient import MyClient
from UserExpand import UserExpandSampler, needed_segments
from utilities import get_user_dict
from utilities import bearer_token
from utilities import get_user_followers_list_by_id, get_user_following_list_by_id

import json
import random
import os
import tqdm

from tweepy.errors import (
    BadRequest, Forbidden, HTTPException, TooManyRequests, TwitterServerError,
    Unauthorized
)

if __name__ == '__main__':
    random.seed(20200120)
    client = MyClient(bearer_token=bearer_token,
                      wait_on_rate_limit=True,
                      proxy='127.0.0.1:15236')
    user_sampler = UserExpandSampler(segments=needed_segments,
                                     top_k=2,
                                     bottom_k=2,
                                     media_k=2,
                                     radius_k=3,
                                     true_k=3,
                                     false_k=3,
                                     sample_k=6
                                     )
    done_ids = json.load(open('user_net/done_user.json'))
    done_ids = set(done_ids)
    while True:
        file_list = os.listdir('user_net/expanding')
        file_list = sorted(file_list, key=lambda x: os.path.getmtime(os.path.join('user_net/expanding', x)))
        expanding_ids = [item.replace('.json', '') for item in file_list]
        for uid in tqdm.tqdm(expanding_ids):
            if uid in done_ids:
                continue
            user = json.load(open('user_net/expanding/' + uid + '.json'))
            try:
                followers = get_user_followers_list_by_id(client, uid)
            except TwitterServerError:
                followers = []
            try:
                following = get_user_following_list_by_id(client, uid)
            except TwitterServerError:
                following = []
            followers_dict = [get_user_dict(item) for item in followers]
            following_dict = [get_user_dict(item) for item in following]
            json.dump(following_dict, open('user_net/following/' + uid + '.json', 'w'), indent=4)
            json.dump(followers_dict, open('user_net/followers/' + uid + '.json', 'w'), indent=4)
            followers_expand = user_sampler.get_expand_users(followers_dict, user)
            following_expand = user_sampler.get_expand_users(following_dict, user)
            for item in following_expand + followers_expand:
                json.dump(item, open('user_net/expanding/' + str(item['id']) + '.json', 'w'), indent=4)
            done_ids.add(uid)
            json.dump([item for item in done_ids], open('user_net/done_user.json', 'w'), indent=4)
