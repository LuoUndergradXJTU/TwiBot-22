import tweepy
import datetime
import time
# TwitterAPI 6253282
# TwitterDev 2244994945

user_fields = ['created_at', 'description', 'entities', 'id', 'location', 'name',
               'pinned_tweet_id', 'profile_image_url', 'protected', 'public_metrics',
               'url', 'username', 'verified', 'withheld']
tweet_fields = ['attachments', 'author_id', 'context_annotations',
                'conversation_id', 'created_at', 'entities', 'geo',
                'id', 'in_reply_to_user_id', 'lang',
                'possibly_sensitive',
                'public_metrics', 'referenced_tweets', 'reply_settings',
                'source', 'text', 'withheld']
# space_fields = ['created_at', 'creator_id', 'ended_at',
#                 'host_ids', 'id', 'invited_user_ids',
#                 'is_ticketed', 'lang', 'participant_count',
#                 'scheduled_start', 'speaker_ids', 'started_at',
#                 'state', 'subscriber_count', 'title', 'topic_ids', 'updated_at']
list_fields = ["id", "name", "created_at", "description", "follower_count",
               "member_count", "private", "owner_id"]
get_tweet_expansions = ['author_id', 'referenced_tweets.id',
                        'referenced_tweets.id.author_id', 'entities.mentions.username',
                        'attachments.poll_ids', 'attachments.media_keys',
                        'in_reply_to_user_id', 'geo.place_id']
get_user_expansions = ['pinned_tweet_id']

bearer_token = 'your_Twitter_API_bearer_token'


def get_user_dict(user):
    if not isinstance(user, tweepy.user.User):
        raise Exception('save_user can only save the Twitter user data')
    data = {}
    for key in user_fields:
        if key not in user:
            data[key] = None
            continue
        if isinstance(user[key], datetime.datetime):
            data[key] = str(user[key])
            continue
        data[key] = user[key]
    assert len(data) == len(user_fields)
    return data


def get_tweet_dict(tweet):
    if not isinstance(tweet, tweepy.tweet.Tweet):
        raise Exception('save_tweet can only save the Twitter tweet data')
    data = {}
    for key in tweet_fields:
        if key not in tweet:
            data[key] = None
            continue
        if isinstance(tweet[key], datetime.datetime):
            data[key] = str(tweet[key])
            continue
        if key == 'referenced_tweets':
            data[key] = []
            if tweet[key] is None:
                data[key] = None
                continue
            for item in tweet[key]:
                data[key].append(item['data'])
            continue
        data[key] = tweet[key]
    assert len(data) == len(tweet_fields)
    return data


def get_user_followers_list_by_id(client, user_id, pages=1):
    followers_list = []
    pagination_token = None
    for i in range(pages):
        data = client.get_users_followers(id=user_id, max_results=1000,
                                          pagination_token=pagination_token,
                                          user_fields=user_fields,
                                          tweet_fields=tweet_fields)
        if data.data is None:
            return []
        followers_list += data.data
        if 'next_token' not in data.meta:
            break
        pagination_token = data.meta['next_token']
    return followers_list


def get_user_following_list_by_id(client, user_id, pages=1):
    following_list = []
    pagination_token = None
    for i in range(pages):
        data = client.get_users_following(id=user_id, max_results=1000,
                                          pagination_token=pagination_token,
                                          user_fields=user_fields,
                                          tweet_fields=tweet_fields)
        if data.data is None:
            return []
        following_list += data.data
        if 'next_token' not in data.meta:
            break
        pagination_token = data.meta['next_token']
    return following_list


def get_user_timeline_by_id(client, user_id, pages=10):
    timeline = []
    pagination_token = None
    for i in range(pages):
        data = client.get_users_tweets(id=user_id, max_results=100,
                                       pagination_token=pagination_token,
                                       user_fields=user_fields,
                                       tweet_fields=tweet_fields,)
        if data.data is None:
            return []
        timeline += data.data
        if 'next_token' not in data.meta:
            break
        pagination_token = data.meta['next_token']
    return timeline


segments = ['attachments', 'author_id', 'context_annotations',
            'conversation_id', 'created_at', 'entities', 'geo',
            'id', 'in_reply_to_user_id', 'lang', 'possibly_sensitive',
            'public_metrics', 'referenced_tweets', 'reply_settings',
            'source', 'text', 'withheld']


def v1tov2(v1data):
    v2data = {}
    for key in segments:
        if key == 'attachments' or \
                key == 'context_annotations' or \
                key == 'referenced_tweets' or \
                key == 'reply_settings' or \
                key == 'withheld':
            v2data[key] = None
        elif key == 'author_id':
            v2data[key] = v1data['user']['id']
        elif key == 'conversation_id':
            v2data[key] = v1data['id']
        elif key == 'possibly_sensitive':
            if key in v1data:
                v2data[key] = v1data[key]
            else:
                v2data[key] = False
        elif key == 'public_metrics':
            v2data[key] = {'retweet_count': v1data['retweet_count'],
                           'reply_count': None,
                           'like_count': v1data['favorite_count'],
                           'quote_count': None}
        elif key == 'text':
            v2data[key] = v1data['full_text']
        elif key == 'created_at':
            time_array = time.strptime(v1data['created_at'], '%a %b %d %H:%M:%S %z %Y')
            other_time = time.strftime('%Y-%m-%d %H:%M:%S+00:00', time_array)
            v2data[key] = str(other_time)
        else:
            v2data[key] = v1data[key]
    return v2data
