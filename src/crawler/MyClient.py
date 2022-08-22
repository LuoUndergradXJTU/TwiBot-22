from tweepy import Client

from collections import namedtuple
import datetime
import logging
from platform import python_version
import time
import warnings

import requests

import tweepy
from tweepy.auth import OAuthHandler
from tweepy.errors import (
    BadRequest, Forbidden, HTTPException, TooManyRequests, TwitterServerError,
    Unauthorized
)
from tweepy.list import List
from tweepy.media import Media
from tweepy.place import Place
from tweepy.poll import Poll
from tweepy.space import Space
from tweepy.tweet import Tweet
from tweepy.user import User

log = logging.getLogger(__name__)

Response = namedtuple("Response", ("data", "includes", "errors", "meta"))


class MyClient(Client):
    def __init__(
        self, bearer_token=None, consumer_key=None, consumer_secret=None,
        access_token=None, access_token_secret=None, *, return_type=Response,
        wait_on_rate_limit=False, proxy=None
    ):
        super().__init__()
        self.bearer_token = bearer_token
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret

        self.return_type = return_type
        self.wait_on_rate_limit = wait_on_rate_limit

        self.session = requests.Session()
        self.user_agent = (
            f"Python/{python_version()} "
            f"Requests/{requests.__version__} "
            f"Tweepy/{tweepy.__version__}"
        )
        self.proxy = {}
        if proxy is not None:
            self.proxy['https'] = proxy

    def request(self, method, route, params=None, json=None, user_auth=False):
        host = "https://api.twitter.com"
        headers = {"User-Agent": self.user_agent}
        auth = None
        if user_auth:
            auth = OAuthHandler(self.consumer_key, self.consumer_secret)
            auth.set_access_token(self.access_token, self.access_token_secret)
            auth = auth.apply_auth()
        else:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        log.debug(
            f"Making API request: {method} {host + route}\n"
            f"Parameters: {params}\n"
            f"Headers: {headers}\n"
            f"Body: {json}"
        )
        # print(params)
        with self.session.request(
            method, host + route, params=params, json=json, headers=headers,
            auth=auth, proxies=self.proxy
        ) as response:
            log.debug(
                "Received API response: "
                f"{response.status_code} {response.reason}\n"
                f"Headers: {response.headers}\n"
                f"Content: {response.content}"
            )

            if response.status_code == 400:
                raise BadRequest(response)
            if response.status_code == 401:
                raise Unauthorized(response)
            if response.status_code == 403:
                raise Forbidden(response)
            # Handle 404?
            if response.status_code == 429:
                if self.wait_on_rate_limit:
                    reset_time = int(response.headers["x-rate-limit-reset"])
                    sleep_time = reset_time - int(time.time()) + 1
                    if sleep_time > 0:
                        log.warning(
                            "Rate limit exceeded. "
                            f"Sleeping for {sleep_time} seconds."
                        )
                        time.sleep(sleep_time)
                    return self.request(method, route, params, json, user_auth)
                else:
                    raise TooManyRequests(response)
            if response.status_code >= 500:
                raise TwitterServerError(response)
            if not 200 <= response.status_code < 300:
                raise HTTPException(response)
            return response