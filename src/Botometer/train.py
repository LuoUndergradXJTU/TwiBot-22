import json
from tqdm import tqdm
from MyBotometer import Botometer, NoTimelineError
from tweepy.error import TweepError
import tweepy
from argparse import ArgumentParser
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--proxy', type=str)
    args = parser.parse_args()
    proxy = args.proxy
    username_path = 'tmp/username/{}'.format(args.dataset_name)
    if not os.path.exists(username_path):
        raise ValueError
    key = json.load(open('tmp/key.json'))
    rapid_api_key = key['rapid_api_key']
    consumer_key = key['consumer_key']
    consumer_secret = key['consumer_secret']
    access_token = key['access_token']
    access_token_secret = key['access_token_secret']
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth,
                     wait_on_rate_limit=True,
                     proxy=proxy,
                     parser=tweepy.parsers.JSONParser())
    bom = Botometer(rapid_api_key=rapid_api_key,
                    twitter_api=api,
                    proxy=proxy)

    usernames = json.load(open(os.path.join(username_path, 'users_test.json')))
    pbar = tqdm(total=len(usernames), ncols=0)
    save_path = 'tmp/scores/{}'.format(args.dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for item in usernames:
        path = os.path.join(save_path, '{}.json'.format(item['id']))
        if os.path.exists(path):
            pbar.update()
            continue
        if item['username'] is None:
            result = 'the username is None'
            json.dump(result, open(path, 'w'))
            pbar.update()
            continue
        username = '@{}'.format(item['username'].strip())
        pbar.set_postfix_str(username)
        try:
            result = bom.check_account(username)
        except NoTimelineError:
            result = 'this user does not have any tweets'
        except TweepError:
            result = 'api can\'t get the timeline'
        json.dump(result, open(path, 'w'))
        pbar.update()


