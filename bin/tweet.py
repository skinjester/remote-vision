#!/usr/bin/python
__author__ = 'Gary Boodhoo'

import tweepy,sys,os,time



def tweet():
    consumer_key='3iSUitN4D5Fi52fgmF5zMQodc'
    consumer_secret='Kj9biRwpjCBGQOmYJXd9xV4ni68IO99gZT2HfdHv86HuPhx5Mq'
    access_key='15870561-2SH025poSRlXyzAGc1YyrL8EDgD5O24docOjlyW5O'
    access_secret='qwuc8aa6cpRRKXxMObpaNhtpXAiDm6g2LFfzWhSjv6r8H'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    fn = os.path.abspath('../1280x720.jpg')
    myStatusText = '@username #deepdreamvisionquest #GGG2016'
    #api.update_status(status=myStatusText)
    api.update_with_media(fn, status=myStatusText )
    
'''
#getting the parameter passed via the shell command from the Arduino Sketch
status = status = sys.argv[1] 
fn = os.path.abspath('/mnt/sda1/moneyplant.png')
#UpdateStatus of twitter called with the image file
'''

# -------
# MAIN
# ------- 
def main():
    print 'welcome to the TweetBot'
    while True:
        tweettoTwitter()
        time.sleep(60)


# -------- 
# INIT
# --------
if __name__ == "__main__":
    main()
