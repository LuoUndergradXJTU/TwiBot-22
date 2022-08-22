import os

from MyClient import MyClient
from utilities import bearer_token
from utilities import get_user_timeline_by_id
from utilities import get_tweet_dict
import random
import tqdm
import json
from tweepy.errors import (
    BadRequest, Forbidden, HTTPException, TooManyRequests, TwitterServerError,
    Unauthorized
)
# 138840988
if __name__ == '__main__':
    random.seed(20200120)
    client = MyClient(bearer_token=bearer_token,
                      wait_on_rate_limit=True,
                      proxy='127.0.0.1:15236')
    crawling_ids = [item.replace('.json', '') for item in os.listdir('user_net/bone')]
    crawling_len = len(crawling_ids)
    for uid in tqdm.tqdm(crawling_ids):
        filepath = 'timeline/' + uid + '.json'
        if os.path.exists(filepath):
            continue
        try:
            tweet_list = get_user_timeline_by_id(client, uid)
        except TwitterServerError:
            tweet_list = []
        tweet_list_dict = [get_tweet_dict(item) for item in tweet_list]
        json.dump(tweet_list_dict, open(filepath, 'w'), indent=4)
