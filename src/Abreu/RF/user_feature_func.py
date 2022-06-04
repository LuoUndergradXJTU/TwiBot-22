import re
import datetime
import time
import math
import pandas as pd
def get_user_id(user):
    return user['id']

def get_username(user):
    return user['username']
    
    
def get_screen_name(user):
    return user['name']
    
    
def has_description(user):
    des = user['description']
    if des is None or des == ' ':
        return False
    else:
        return True
        
             
def get_statuses(user):
    tweets = user['public_metrics']['tweet_count']
    if tweets is None:
        return 0
    else:
        return int(tweets)
        
               
def get_followers(user):
    followers_count = user['public_metrics']['followers_count']
    if followers_count is None:
        return 0
    else:
        return int(followers_count)
   
def get_friends(user):
    following_count = user['public_metrics']['following_count']
    if following_count is None:
        return 0
    else:
        return int(following_count)   
        
        
def get_favourites(user):
    favourites_count = 0
    return int(favourites_count)      
    
     
def get_lists(user):
    lists = user['public_metrics']['listed_count']
    if lists is None:
        return 0
    else:
        return int(lists)  


def has_default_profile(user):
    if user["profile_image_url"] is None or user["profile_image_url"] == ' ':
        return False
    else:
        return True
          
    
def is_verified(user):
    if user['verified'] == 'True ':
        return True
    else:
        return False
        
def get_freq(user):
    tweets = user['public_metrics']['tweet_count']
    GMT_FORMAT = '%a %b %d %H:%M:%S %z %Y'
    create_time = user['created_at']
    user_create_time = datetime.datetime.strptime(create_time, GMT_FORMAT)
    d1 = user_create_time.date()
    #d1 = datetime.datetime.strptime(user_create_time, "%Y-%m-%d").date()
    present_time = datetime.datetime.now()
    d2 = present_time.date()
    #d2 = datetime.datetime.strptime(present_time, "%Y-%m-%d").date()
    user_age = (d2-d1).days
    freq = int(tweets)/int(user_age)
    if create_time is None or create_time == ' ':
        return False
    else:
        return(freq)


def get_followers_growth(user):
    followers_count = user['public_metrics']['followers_count']
    GMT_FORMAT = '%a %b %d %H:%M:%S %z %Y'
    create_time = user['created_at']
    user_create_time = datetime.datetime.strptime(create_time, GMT_FORMAT)
    d1 = user_create_time.date()
    present_time = datetime.datetime.now()
    d2 = present_time.date()
    user_age = (d2-d1).days
    followers_growth = int(followers_count)/int(user_age)
    return(followers_growth)
    
    
def get_listed_growth(user):
    listed_count = user['public_metrics']['listed_count']
    GMT_FORMAT = '%a %b %d %H:%M:%S %z %Y'
    create_time = user['created_at']
    user_create_time = datetime.datetime.strptime(create_time, GMT_FORMAT)
    d1 = user_create_time.date()
    present_time = datetime.datetime.now()
    d2 = present_time.date()
    user_age = (d2-d1).days
    listed_growth = int(listed_count)/int(user_age)
    return(listed_growth) 
    
    
def get_followers_friends_ratio(user):
    followers_count = user['public_metrics']['followers_count']
    following_count = user['public_metrics']['following_count']
    if following_count == 0:
    #if following_count is None or following_count == '0':
        return 0.0
    else:
        return int(followers_count) / int(following_count)
    
    
def get_screen_name_length(user):
    screen_name = get_screen_name(user)
    if screen_name is None:
        return 0
    else:
        return len(screen_name) - 1    
    
    
def get_num_digits_in_screen_name(user):
    screen_name = get_screen_name(user)
    if screen_name is None:
        return 0
    else:
        numbers = re.findall(r'\d+', screen_name)
        return len(numbers)

def get_name_length(user):
    username = get_username(user)
    if username is None:
        return 0
    else:
        return len(username) - 1


def get_num_digits_in_name(user):
    username = get_username(user)
    if username is None:
        return 0
    else:
        numbers = re.findall(r'\d+', username)
        return len(numbers)


def get_description_length(user):
    if has_description(user):
        return len(user['description'])
    else:
        return 0

def get_screen_name_likelihood(user):
    return 0
    

    
    

    
     
