import os
import sys
import csv
import ast
import tqdm
import tweepy

from tweepy import OAuthHandler

def run(inp, oup, keystring):
    # info = {"CONSUMER_KEY": "", "CONSUMER_SECRET": "", "ACCESS_KEY": "", "ACCESS_SECRET": ""}
    info =  ast.literal_eval(keystring)

    consumer_key = info['CONSUMER_KEY']
    consumer_secret = info['CONSUMER_SECRET']
    access_token = info['ACCESS_KEY']
    access_secret = info['ACCESS_SECRET']

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    with open(inp, "r") as df, open(oup, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["handle", "name", "description", "friends_count", "followers_count", "status", "isDem"])
        for i in tqdm.tqdm(df.readlines()):
            id, kind = i.split("\t")

            # User Info
            kind = 1 if str(kind)[0] == "d" or str(kind)[0] == "D" else 0
            try:
                user_obj = api.get_user(id)
                user_handle = user_obj.screen_name
                user_name = user_obj.name
                user_desc = user_obj.description
                user_friends_count = len(api.friends_ids(id))
                user_followers_count = len(api.followers_ids(id))
            except tweepy.error.TweepError:
                continue

            # Tweets
            tweets = [i.text for i in api.user_timeline(id, count=200)]

            # Compile!
            for tweet in tweets:
                writer.writerow([user_handle, user_name, user_desc, user_friends_count, user_followers_count, tweet, kind])
            csvfile.flush()

    # print("Writing data...")
    # with :
      
