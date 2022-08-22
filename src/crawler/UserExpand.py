import time
import random
import math


# TwitterAPI 6253282
# TwitterDev 2244994945
needed_segments = ['created_at', 'public_metrics.followers_count',
                   'public_metrics.following_count',
                   'public_metrics.tweet_count',
                   'public_metrics.listed_count',
                   'verified',
                   'entities.url']


def sample_from_numerical_data_distribution(data, top_k, bottom_k, media_k):
    data.sort(key=lambda x: x[1])
    data_split_size = math.ceil(len(data) * 0.2)
    data_top = data[:data_split_size]
    data_bottom = data[-data_split_size:]
    data_media = data[data_split_size:-data_split_size]
    top_k = min(len(data_top), top_k)
    bottom_k = min(len(data_bottom), bottom_k)
    media_k = min(len(data_media), media_k)
    sample_data =\
        random.choices(data_top, k=top_k) + \
        random.choices(data_media, k=media_k) + \
        random.choices(data_bottom, k=bottom_k)
    return [item[0] for item in sample_data]


def sample_from_numerical_data_difference(data, value, radius_k):
    data.sort(key=lambda x: x[1])
    loc_index = 0
    if value <= data[0][1]:
        loc_index = 0
    elif value > data[-1][1]:
        loc_index = len(data) - 1
    else:
        for index in range(len(data)-1):
            if data[index][1] < value <= data[index+1][1]:
                loc_index = index
                break
    loc_index = len(data) - loc_index - 1
    loc_l = max(0, loc_index - radius_k)
    loc_r = min(len(data), loc_index + radius_k)
    sample_data = data[loc_l:loc_r]
    return [item[0] for item in sample_data]


def sample_from_boolean_data_distribution(data, true_k, false_k):
    true_data = []
    false_data = []
    for item in data:
        if item[1]:
            true_data.append(item[0])
        else:
            false_data.append(item[0])
    true_k = min(len(true_data), true_k)
    false_k = min(len(false_data), false_k)
    sample_true = random.choices(true_data, k=true_k)
    sample_false = random.choices(false_data, k=false_k)
    return sample_true + sample_false


def sample_from_boolean_data_difference(data, value, sample_k):
    true_data = []
    false_data = []
    for item in data:
        if item[1]:
            true_data.append(item[0])
        else:
            false_data.append(item[0])
    if value:
        sample_k = min(len(false_data), sample_k)
        sample_data = random.choices(false_data, k=sample_k)
    else:
        sample_k = min(len(true_data), sample_k)
        sample_data = random.choices(true_data, k=sample_k)
    return sample_data


def get_segment_data(user, segment):
    segments = segment.split('.')
    if segments[-1] == 'created_at':
        try:
            utime = time.mktime(time.strptime(user['created_at'], '%Y-%m-%d %H:%M:%S%z'))
        except Exception as e:
            print(e)
            utime = 0.0
        return utime
    if segments[-1] == 'url':
        return not ((user['entities'] is None) or ('url' not in user['entities']))
    data = user
    for item in segments:
        data = data[item]
    return data


def sample_with_segment(data, user, segment, top_k, bottom_k, media_k, false_k, true_k, sample_k, radius_k):
    segment_data = []
    value = get_segment_data(user, segment)
    for index, item in enumerate(data):
        segment_data.append((index, get_segment_data(item, segment)))
    if isinstance(value, bool):
        distribution_sample = sample_from_boolean_data_distribution(segment_data,
                                                                    true_k=true_k,
                                                                    false_k=false_k)
        difference_sample = sample_from_boolean_data_difference(segment_data, value,
                                                                sample_k=sample_k)
    else:
        distribution_sample = sample_from_numerical_data_distribution(segment_data,
                                                                      top_k=top_k,
                                                                      bottom_k=bottom_k,
                                                                      media_k=media_k)
        difference_sample = sample_from_numerical_data_difference(segment_data, value,
                                                                  radius_k=radius_k)
    return distribution_sample + difference_sample


class UserExpandSampler:
    def __init__(self, segments, top_k, bottom_k, media_k, radius_k, true_k, false_k, sample_k):
        self.segments = segments
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.media_k = media_k
        self.radius_k = radius_k
        self.true_k = true_k
        self.false_k = false_k
        self.sample_k = sample_k

    def get_expand_users(self, data, user):
        if len(data) == 0:
            return []
        segment = random.choice(self.segments)
        sample_ids = sample_with_segment(data, user, segment=segment,
                                         top_k=self.top_k,
                                         bottom_k=self.bottom_k,
                                         media_k=self.media_k,
                                         radius_k=self.radius_k,
                                         true_k=self.true_k,
                                         false_k=self.false_k,
                                         sample_k=self.sample_k)
        sample_ids = set(sample_ids)
        return [data[item] for item in sample_ids]
