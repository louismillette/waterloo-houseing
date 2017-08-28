import re
import sqlite3
import facebook
import os
import dateutil.parser
import datetime
import time
import json
import requests
from statistics import median
import math
import os
import random
from sklearn import tree
import numpy as np
import pydotplus
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import treebank

import string
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split as tts
from twilio.rest import Client

# +17055232728 <- twilo phone number (canada)

## Credentials

_to_phone_number = 'PROTECTED_CREDENTIAL'
_from_phone_number = 'PROTECTED_CREDENTIAL'
_account_sid = 'PROTECTED_CREDENTIAL'
_auth_token = 'PROTECTED_CREDENTIAL'
_facebook_app_id = 'PROTECTED_CREDENTIAL'
_facebook_api_version = 'PROTECTED_CREDENTIAL'
_facebook_app_secret = 'PROTECTED_CREDENTIAL'
_facebook_token = 'PROTECTED_CREDENTIAL'
_facebook_group_ID = 'PROTECTED_CREDENTIAL' #waterloo textbook group id

# https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
class NLTKPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, punct=None,
                 lower=True, strip=True):
        self.lower = lower
        self.strip = strip
        self.stopwords = stopwords or set(sw.words('english'))
        self.punct = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

# lose collection of emthods for creating the DB and handleing messages
class Catigorize():
    def __init__(self):
        self.facebookAppId = _facebook_app_id
        self.facebookApiVersion = _facebook_api_version
        self.facebookAppSecret = _facebook_app_secret
        self.facebookToken = _facebook_token
        #expires Never, 2017.  Get new one at https://developers.facebook.com/tools/accesstoken/
        self.facebookGroupID = _facebook_group_ID
        self.dbsetup()

    # pull in (num * pages) posts from the facebookGroupID page id
    # if start is given, starts by pulling from given start url, instead of from the regular starting endpoint
    def GroupFeed(self, num=5, pages=5, start=None, since=None, until=None):
        baseurl = "https://graph.facebook.com/v2.10/{}/feed?".format(self.facebookGroupID)
        if start:
            group = json.loads(requests.get(start).text)
            try:
                group['paging']['next']
            except:
                return group
        elif since and until:
            since = "{}-{}-{}".format(since.year,since.month,since.day)
            until = "{}-{}-{}".format(until.year, until.month, until.day)
            # graph = facebook.GraphAPI(access_token=self.facebookToken, version=self.facebookApiVersion)
            url = "{}{}&since={}&until={}&access_token={}".format(
                baseurl,
                'fields=created_time,from,message',
                since,until,
                self.facebookToken
            )
            group = json.loads(requests.get(url).text)
        else:
            url = "{}{}&limit={}&{}&access_token={}".format(
                baseurl,
                'fields=created_time,from,message,comments',
                num,
                "use_actual_created_time_for_backdated_post=true",
                self.facebookToken
            )
            group = json.loads(requests.get(url).text)
        while pages > 0:
            group_next = json.loads(requests.get(group['paging']['next']).text)
            group['data'] += group_next['data']
            try:
                group['paging']['next'] = group_next['paging']['next']
            except:
                return group
            pages -= 1
        return group

    # determines if message if for someone buying, selling, or lease takovers (which includes selling rentals or anything for 12 months)
    # returns b,s, or l for buying selling or lease takeovers, as F or M for weather or not the post is for females only, or mixed gender
    def rank(self,message):
        message = message.lower().replace('\n',' ')
        gender_tag = 'F' if 'female' in message else 'M'
        location_tag = 'T' if 'toronto' in message else 'W'
        buying = [
            r'looking to (rent|lease|sublet)',
            r'looking (for|4)[\s]?[:]?[-]?[a]?[n]?[\s]?(4|8|12|four|eight|twelve)?([\s]?(or|-)[\s]?([1-9]|one|two|three|four|five))?(\s|-)(month[s]?)?[\s]?(sublet|lease)',
            r'looking (for|4)[\s]?[:]?[-]?[a]?[n]?[\s]?(fall|winter|spring|summer) (2015 |2016 |2017 )?sublet',
            r'looking (for|4)[\s]?[:]?[-]?[a]?[n]?[\s]?([\s]?entire[\s]?)?(house|room|quad|place)',
            r'looking (for|4)[\s]?[:]?[-]?[a]?[n]?[\s]?([1-9]|one|two|three|four|five)([\s]?(or|-)[\s]?([1-9]|one|two|three|four|five))?[\s]?(bed)?room[s]?',
            r'looking for[\s]?[a]?[\s]?4(\s|-)month[\s](fall|winter}summer|spring)?sublet',
            r'looking for[\s]?[a]?[\s]?(fall|winter|spring|summer)',
            r'looking for[\s]?[a]?[1-9]?()?[\s]?(room[s]?|sublet[s]?)',
            r'wanted',
            r'looking:',
        ]
        selling = [
            r'lease available',
            r'[0-9]? rooms left',
            r'(sublet[t]?(ing)?|sub(-|\s)?lease?(ing)?|sell(ing)?) (my|a) room',
            r'((bed)?room[s]?|sublet[t]?[s]?|unit[s]?) (for (a )?student[s]? )?available',
            r'available for (the )?(fall|summer|spring|winter)',
            r'(sublet[t]?(ing)?|sub(-|\s)?lease?(ing)?) for (the )?(summer|spring|fall|winter)',
            r'(sublet[t]?(ing)?|sub(-|\s)?lease?(ing)?) ([1-9]|one|two|three|four|five|six) (bed)?room(\')?(s)?',
            r'([1-3]|one|two|three) months (free|off)',
            r'room for rent',
            r'(?!looking[\s]?[-]?for[:]?[\s]?[-]?)(lease|contract)(\s|-)takeover',
            r'(?!(looking[\s]?[-]?for[:]?[\s]?[-]?)|(wanted[:]?[\s]?[-]?))(1 year|12 month[s]?|year|8(\s|-)month[s]?|4(\s|-)month[s]?|four(\s|-)month[s]?) (lease|sublet)',
            # r'12 month lease',
            # r'year lease',
            # r'for rent'
        ]
        buying_score = sum([1 if re.search(ele,message) else 0 for ele in buying])
        selling_score = sum([1 if re.search(ele,message) else 0 for ele in selling])
        if buying_score > selling_score:
            return 'b',gender_tag
        elif selling_score > buying_score:
            return 's',gender_tag
        elif buying_score != 0 and buying_score == selling_score:
            return 'b', gender_tag
        else:
            # lets try some looser identifiers
            selling = [
                r'looking to (sublet|lease)',
                r'lease term',
                r'12 month lease',
                r'year lease',
                r'for rent',
                r'lease(-|\s)take( - |-|\s)?over',
                r'(?!looking[\s]?[-]?for[:]?[\s]?[-]?[a]?)(summer|spring|fall|winter) sublet ',
                r'(unit[s]?|suite[s]?) (left|remaining)',
            ]
            buying = [
                r'looking for',
            ]
            selling_score = sum([1 if re.search(ele, message) else 0 for ele in selling])
            buying_score = sum([1 if re.search(ele, message) else 0 for ele in buying])
            if selling_score >= buying_score and selling_score != 0:
                return 's', gender_tag
            if buying_score == selling_score and buying_score != 0:
                return 'b', gender_tag
            else:
                return None,gender_tag

    # produces the matrix of expression usages
    def rankmatrix(self,message):
        reducedMessage=[]
        exprs = [
            r'looking to (rent|lease|sublet)',
            r'looking (for|4)[\s]?[:]?[-]?[a]?[n]?[\s]?(4|8|12|four|eight|twelve)?([\s]?(or|-)[\s]?([1-9]|one|two|three|four|five))?(\s|-)(month[s]?)?[\s]?(sublet|lease)',
            r'looking (for|4)[\s]?[:]?[-]?[a]?[n]?[\s]?(fall|winter|spring|summer) (2015 |2016 |2017 )?sublet',
            r'looking (for|4)[\s]?[:]?[-]?[a]?[n]?[\s]?([\s]?entire[\s]?)?(house|room|quad|place)',
            r'looking (for|4)[\s]?[:]?[-]?[a]?[n]?[\s]?([1-9]|one|two|three|four|five)([\s]?(or|-)[\s]?([1-9]|one|two|three|four|five))?[\s]?(bed)?room[s]?',
            r'looking for[\s]?[a]?[\s]?4(\s|-)month[\s](fall|winter}summer|spring)?sublet',
            r'looking for[\s]?[a]?[\s]?(fall|winter|spring|summer)',
            r'looking for[\s]?[a]?[1-9]?()?[\s]?(room[s]?|sublet[s]?)',
            r'wanted',
            r'looking:',
            r'lease available',
            r'[0-9]? rooms left',
            r'(sublet[t]?(ing)?|sub(-|\s)?lease?(ing)?|sell(ing)?) (my|a) room',
            r'((bed)?room[s]?|sublet[t]?[s]?|unit[s]?) (for (a )?student[s]? )?available',
            r'available for (the )?(fall|summer|spring|winter)',
            r'(sublet[t]?(ing)?|sub(-|\s)?lease?(ing)?) for (the )?(summer|spring|fall|winter)',
            r'(sublet[t]?(ing)?|sub(-|\s)?lease?(ing)?) ([1-9]|one|two|three|four|five|six) (bed)?room(\')?(s)?',
            r'([1-3]|one|two|three) months (free|off)',
            r'room for rent',
            r'(?!looking[\s]?[-]?for[:]?[\s]?[-]?)(lease|contract)(\s|-)takeover',
            r'(?!(looking[\s]?[-]?for[:]?[\s]?[-]?)|(wanted[:]?[\s]?[-]?))(1 year|12 month[s]?|year|8(\s|-)month[s]?|4(\s|-)month[s]?|four(\s|-)month[s]?) (lease|sublet)',
        ]
        for expr in exprs:
            reducedMessage.append(1 if reg else 0)
            # reducedMessage.append(2 if rank == 'b' else( 1 if rank == 's' else 0))
        return reducedMessage

    # attempts to catagorize period given message.
    # optional priority for when theres a tie with first given most and last given least
    def period(self,message, priority=[4,8,12]):
        message = message.lower().replace('\n','')
        fourmonth = [r'4[\s]?[-]?month[s]?',r'four[\s]?month[s]?',
                     r'(fall|winter|summer|spring) sublet',
                     r'sublet (for)? (fall|winter|summer|spring)',
                     r'(sept(ember)?|sep(tember)?|aug(ust)?)([\s]?2017)?([\s]?2016)?([\s]?2015)?[\s]?[-]?(to)?[\s]?(dec(\.)?(ember)?|jan(\.)?(uary)?)',
                     r'(jan(\.)?(uary)?|dec(\.)?(ember)?)([\s]?2017)?([\s]?2016)?([\s]?2015)?[\s]?[-]?(to)?[\s]?(april|may)',
                     r'(may|mar(\.)?(ch))([\s]?2017)?([\s]?2016)?([\s]?2015)?[\s]?[-]?(to)?[\s]?(aug(\.)?(ust)?|sept(ember)?|sep(tember)?)'
                     ]
        eightmonth = [r'8[\s]?[-]?month[s]?',
                      r'eight[\s]?month[s]?',
                      r'(sept(ember)?|sep(tember)?|aug(ust)?)([\s]?2017)?([\s]?2016)?([\s]?2015)?[\s]?[-]?(to)?[\s]?(april|may)',
                      r'(jan(\.)?(uary)?|dec(\.)?(ember)?)([\s]?2017)?([\s]?2016)?([\s]?2015)?[\s]?[-]?(to)?[\s]?(aug(\.)?(ust)?|sept(ember)?|sep(tember)?)',
                      r'(may|mar(\.)?(ch))([\s]?2017)?([\s]?2016)?([\s]?2015)?[\s]?[-]?(to)?[\s]?(dec(\.)?(ember)?|jan(\.)?(uary)?)'
                      ]
        twelvemonth = [r'12[\s]?[-]?month[s]?',
                       r'tweleve[\s]?month[s]?',
                       r'2017[-]?[to]?[\s]?2018',
                       r'2016[-]?[to]?[\s]?2017',
                       r'2015[-]?[to]?[\s]?2016',
                       r'(one)?(full)?[\s]?year(\'s)?[\s]?(lease)?',
                       r'(sept(ember)?|sep(tember)?|aug(ust)?)([\s]?2017)?([\s]?2016)?([\s]?2015)?[\s]?[-]?(to)?[\s]?(sept(ember)?|sep(tember)?|aug(ust)?)',
                       r'(jan(\.)?(uary)?|dec(\.)?(ember)?)([\s]?2017)?([\s]?2016)?([\s]?2015)?[\s]?[-]?(to)?[\s]?(jan(\.)?(uary)?|dec(\.)?(ember)?)',
                       r'(may|mar(\.)?(ch))([\s]?2017)?([\s]?2016)?([\s]?2015)?[\s]?[-]?(to)?[\s]?(may|mar(\.)?(ch))'
                       ]
        four_score = sum([1 if re.search(ele, message) else 0 for ele in fourmonth])
        eight_score = sum([1 if re.search(ele, message) else 0 for ele in eightmonth])
        tweleve_score = sum([1 if re.search(ele, message) else 0 for ele in twelvemonth])
        if four_score > eight_score >= tweleve_score or four_score > tweleve_score >= eight_score:
            return 4
        elif eight_score > four_score >= tweleve_score or eight_score > tweleve_score >= four_score:
            return 8
        elif tweleve_score > eight_score >= four_score or tweleve_score > four_score >= eight_score:
            return 12
        elif tweleve_score == eight_score and tweleve_score != 0:
            if priority.index(12) > priority.index(8):
                return 12
            else:
                return 8
        elif eight_score == four_score and eight_score != 0:
            if priority.index(8) > priority.index(4):
                return 8
            else:
                return 4
        elif four_score == tweleve_score and tweleve_score != 0:
            if priority.index(12) > priority.index(4):
                return 12
            else:
                return 4
        else:
            # if Nothing else, lets look at these looser indicators of period
            fourmonth = [r'(fall|winter|spring|summer) (2015|2016|2017|2018)?']
            twelvemonth = [r'for rent',
                           r'school year',
                           r'lease(\s|-)take[\s]?[-]?over',
                           r'take[\s]?[-]?over (a )?lease']
            four_score = sum([1 if re.search(ele, message) else 0 for ele in fourmonth])
            tweleve_score = sum([1 if re.search(ele, message) else 0 for ele in twelvemonth])
            if four_score > tweleve_score:
                return 4
            elif tweleve_score > four_score:
                return 12
            elif four_score == tweleve_score and four_score != 0:
                return 12
            else:
                return None

    # if DB does not create it.  If table posts does not exist, create it
    def dbsetup(self):
        # creates this if it doen not already exist
        conn = sqlite3.connect('Sublets.db')
        c = conn.cursor()
        try:
            # Create table
            c.execute('''CREATE TABLE posts
                         (date text, name text, term text, post text, gender text, action text, period text, location text, price real)''')
        except:
            pass # table already exists
        conn.commit()
        conn.close()

    # where is this place being sold?
    def location(self,message):
        message = message.replace('\n',' ').lower()
        streets = ['king', 'columbia', 'lester', 'albert', 'phillip', 'hickory', 'sunview', 'hemlock', 'ezra', 'bricker', 'spruce', 'maple', 'state', 'regina',
                   'holly', 'hazel', 'westmount', 'westcourt', 'keats way', 'shakespear', 'mcdougall', 'erb', 'drummerhil', 'cardill', 'cedarbrae', 'weber',
                   'helene', 'marshall', 'midwood', 'old albert', 'univerisity street', 'univerisity road']
        buildings = ['icon', 'blair house', 'preston', 'lux', 'luxe', 'bridgeport house', 'sage', 'wcri', 'king street towers']
        adress = None

        for street in streets:
            # {{}} because python .format is applied
            m = re.search(r'([^a-zA-Z])\s?([0-9]{{2,3}}[a-zA-Z]|[0-9]{{0,3}})\s?{}([^a-zA-Z])'.format(street), message)
            if m:
                adress = '{} {}'.format(m.group(2), street) if m.group(2) != '' else street
        for building in buildings:
            m = re.search(r'[^a-zA-Z]{}[^a-zA-Z]'.format(building), message)
            if m:
                adress = building
        return adress

    # what price is the person looking for, or asking?
    def price(self,message):
        regex = r'\$[0-9]?,?[0-9]{3}'
        hits = re.findall(regex,message)
        hits = [int(ele.replace(',','')[1:]) for ele in hits]
        # we want monthly price, reasonable for a student.  Offers outside this are not worth considering
        hits = list(filter(lambda x: 1500 > x > 240, hits))
        if len(hits) > 1:
            return median(hits)
        if len(hits) == 0:
            return None
        return hits[0]
        # price is still ambiguous

    # get the term from the date
    def schoolterm(self,date):
        dt = dateutil.parser.parse(date)
        dateOw = [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second]
        term = '1' + str(dateOw[0] % 1000) + (
        '1' if (dateOw[1] < 3 or dateOw[1] > 10) else ('5' if (dateOw[1] > 2 and dateOw[1] < 7) else '9'))
        if dateOw[1] == 12:
            # ther current year is off by one
            term = term[0] + str(int(term[1:3]) + 1) + term[3]
        return term

    # gets the data, appropriatedly useing all function methods to get, catagorize, tag, and store posts
    # datapoints is the number of desired data points we want to fetch
    # DO NOT run this twice, without clearing the DB first.  It will add repeat entries
    def getdata(self,datapoints=10000):
        num = 50
        pages = round(datapoints/50)
        group = self.GroupFeed(num,1)
        conn = sqlite3.connect('Sublets.db')
        c = conn.cursor()
        # we'll load fb posts and process them 50 at a time, to avoid mem. overflow
        while pages > 1:
            for data in group['data']:
                try:
                    data['message']
                except:
                    continue
                # our well defined constraint
                c.execute('INSERT INTO posts VALUES (?,?,?,?,?,?,?,?,?)', [
                    data['created_time'],
                    data['from']['name'],
                    self.schoolterm(data['created_time']),
                    data['message'],
                    self.rank(data['message'])[1],
                    self.rank(data['message'])[0],
                    self.period(data['message']),
                    self.location(data['message']),
                    self.price(data['message'])
                ])
            conn.commit()
            next = group['paging']['next']
            group = self.GroupFeed(num=50,pages=1,start=next)
            pages -= 1
        conn.close()

    # take our db data and put it in (readable!) CSV
    def tocsv(self,name='data.csv'):
        conn = sqlite3.connect('Sublets.db')
        c = conn.cursor()
        labels = ['date','name','term','post','gender','action','period','location','price']
        scriptdir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(scriptdir,name),'w') as file:
            file.write(','.join(labels) + '\n')
            for row in c.execute('SELECT * FROM posts'):
                row = [str(str(ele).encode('Ascii',errors='replace'), 'utf-8').replace(',','').replace('\n','') for ele in row]
                file.write(','.join(row) + '\n')

    #delete repeats in perfect table.  DB and table must already exists or it'll complain
    def dedlrepeats(self):
        conn = sqlite3.connect('Sublets.db')
        c = conn.cursor()
        datarows = list(c.execute('SELECT * FROM perfect'))
        norepeats = []
        norepeatids = []
        for row in datarows:
            if (row[1],row[2]) in norepeatids:
                continue
            else:
                norepeats.append(row)
                norepeatids.append((row[1],row[2]))

        c.execute('DROP TABLE perfect;')
        c.execute('''CREATE TABLE perfect(date text, name text, term text, post text, gender text, action text, period text, location text, price real)''')
        for row in norepeats:
            c.execute('INSERT INTO perfect VALUES (?,?,?,?,?,?,?,?,?)', [
                row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8]
            ])
        conn.commit()
        conn.close()

# ML algorithum we'll use to improve guessing weahter or not someone is buying or selling
class Classify(Catigorize, NLTKPreprocessor):
    def __init__(self):
        Catigorize.__init__(self)
        NLTKPreprocessor.__init__(self)

    def build(self,X, Y=None, classifier=SGDClassifier):
        if isinstance(classifier, type):
            classifier = classifier()

        model = Pipeline([
            ('preprocessor', NLTKPreprocessor()),
            ('vectorizer', TfidfVectorizer(
                tokenizer=lambda x: x, preprocessor=None, lowercase=False
            )),
            ('classifier', classifier),
        ])

        model.fit(X, Y)
        return model

    # reduces text into array of numbers based on which expressions are matched in the message
    def terminology(self,message,period):
        reducedMessage = []
        exprs = [
            r'lease available',
            r'[0-9]? rooms left',
            r'(sublet[t]?(ing)?|sub(-|\s)?lease?(ing)?|sell(ing)?) (my|a) room',
            r'((bed)?room[s]?|sublet[t]?[s]?|unit[s]?) (for (a )?student[s]? )?available',
            r'available for (the )?(fall|summer|spring|winter)',
            r'(sublet[t]?(ing)?|sub(-|\s)?lease?(ing)?) for (the )?(summer|spring|fall|winter)',
            r'(sublet[t]?(ing)?|sub(-|\s)?lease?(ing)?) ([1-9]|one|two|three|four|five|six) (bed)?room(\')?(s)?',
            r'([1-3]|one|two|three) months (free|off)',
            r'room for rent',
            r'(?!looking[\s]?[-]?for[:]?[\s]?[-]?)(lease|contract)(\s|-)takeover',
            r'(?!(looking[\s]?[-]?for[:]?[\s]?[-]?)|(wanted[:]?[\s]?[-]?))(1 year|12 month[s]?|year|8(\s|-)month[s]?|4(\s|-)month[s]?|four(\s|-)month[s]?) (lease|sublet)',
            r'lease available',
            r'[0-9]? rooms left',
            r'(sublet[t]?(ing)?|sub(-|\s)?lease?(ing)?|sell(ing)?) (my|a) room',
            r'((bed)?room[s]?|sublet[t]?[s]?|unit[s]?) (for (a )?student[s]? )?available',
            r'available for (the )?(fall|summer|spring|winter)',
            r'(sublet[t]?(ing)?|sub(-|\s)?lease?(ing)?) for (the )?(summer|spring|fall|winter)',
            r'(sublet[t]?(ing)?|sub(-|\s)?lease?(ing)?) ([1-9]|one|two|three|four|five|six) (bed)?room(\')?(s)?',
            r'([1-3]|one|two|three) months (free|off)',
            r'room for rent',
            r'(?!looking[\s]?[-]?for[:]?[\s]?[-]?)(lease|contract)(\s|-)takeover',
            r'(?!(looking[\s]?[-]?for[:]?[\s]?[-]?)|(wanted[:]?[\s]?[-]?))(1 year|12 month[s]?|year|8(\s|-)month[s]?|4(\s|-)month[s]?|four(\s|-)month[s]?) (lease|sublet)',
        ]
        for expr in exprs:
            reg = re.findall(expr, message)
            reducedMessage.append(len(reg))
            reducedMessage.append(period)
            rank = self.rank(message)[0]
            # reducedMessage.append(2 if rank == 'b' else( 1 if rank == 's' else 0))
        return np.array(reducedMessage)

    # takes X in the form output by terminology (except, as a list of such values),
    # takes Y as a list of 0,1, or 2 (0 for buy, 1 for sell, 2 for None)
    # ONLY uses actual post, not anything else, to determine if the post is a buy sell or NA
    # returns clf class and outputs decision tree as PDF
    def train(self,X,Y):
        clf = tree.RandomForestClassifier(n_estimators=47)
        clf = clf.fit(X, Y)
        # dot_data = tree.export_graphviz(clf, out_file=None)
        # graph = pydotplus.graph_from_dot_data(dot_data)
        # graph.write_pdf("traind_set.pdf")
        return clf

    # uses row from DB, graphs fitted v actual values for each aux car (month, term), and spits out success prob.
    # assumes rows are agtherd as a list of tuples: (date,name,term,post,gender,action,period,location,price)
    def graphTest(self,datarows_fitted, datarows_perfect):
        allofit = [list(ele) for ele in zip(datarows_fitted, datarows_perfect)]
        print('all of it length: {}'.format(len(allofit)))
        graph_data = {
            # 'Total':allofit,
            'Four Months':list(filter(lambda x: x[1][6] == '4', allofit)),
            'Eight Months': list(filter(lambda x: x[1][6] == '8', allofit)),
            'TwelveMonths': list(filter(lambda x: x[1][6] == '12', allofit)),
            'Well Defined Sales Accuracy': list(filter(lambda x: x[0][5] == 's' and x[0][6] and x[0][7] and x[0][8], allofit)),
            'Well Defined Sales Coverage': list(filter(lambda x: x[1][5] == 's' and x[0][6] and x[0][7] and x[0][8], allofit)),
            'Overall': allofit,
        }
        plot_index = 1
        for key in graph_data.keys():
            data = graph_data[key]
            X = [ele[0][5] for ele in data]
            Y = [ele[1][5] for ele in data]
            Z = [ele[0]==ele[1] for ele in zip(X,Y)]
            # Plot the decision boundary
            a = plt.subplot(2, 3, plot_index)
            plot_index += 1
            try:
                plt.xlabel('Success,Failure -- {}% correct'.format(str(100*float(sum(Z)/len(Z)))[:4]))
            except:
                plt.xlabel('Success,Failure -- NA')
            plt.ylabel('Number')
            a.axis('tight')
            a.bar([0,1], [sum(Z), len(Z)-sum(Z)])
            a.axis("tight")
            a.set_title(key)
        plt.suptitle("Acurately vs Erroniously Labeld Posts, by Month Term")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    def loadperfect(self):
        conn = sqlite3.connect('Sublets.db')
        c = conn.cursor()
        perfect = c.execute('SELECT * FROM perfect ORDER BY date')
        return list(perfect)

    def loadnomachine(self):
        conn = sqlite3.connect('Sublets.db')
        c = conn.cursor()
        data = [list(ele) for ele in c.execute('SELECT * FROM perfect ORDER BY date')]
        gen_data = [ele[:5] + [self.rank(ele[3])[0]] + ele[6:] for ele in data]
        return gen_data

    # generate the machine learning rows (returns predicted rows, then perfect rows)
    def createMLrows(self, split=.7, method='text'):
        data = self.loadperfect()
        # split perfect into training and testing data
        random.shuffle(data)
        train = data[:int(len(data) * split)]
        perfect_test = data[int(len(data) * split):]
        generated_test = data[int(len(data) * (1-split)):]
        action_train = [ele[5] for ele in train]
        # the array of terms usage for each message
        Y_train = [0 if ele == 'b' else (1 if ele == 's' else 2) for ele in action_train]
        if method == 'expr':
            X_train = [self.terminology(ele[3],ele[6] if ele[6] else 0) for ele in train]
            X_test = [self.terminology(ele[3],ele[6] if ele[6] else 0) for ele in generated_test]
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(X=X_train, y=Y_train)
        else:
            X_train = [ele[3] for ele in train]
            X_test = [ele[3] for ele in generated_test]
            clf = self.build(X=X_train, Y=Y_train)
        pred_values = clf.predict(X_test)
        pred_values_len = len(pred_values)
        i = 0
        while i < pred_values_len:
            if pred_values[i] == 0:
                lst = list(generated_test[i])
                lst[5] = 'b'
                generated_test[i] = tuple(lst)
            elif pred_values[i] == 1:
                lst = list(generated_test[i])
                lst[5] = 's'
                generated_test[i] = tuple(lst)
            elif pred_values[i] == 2:
                lst = list(generated_test[i])
                lst[5] = None
                generated_test[i] = tuple(lst)
            else:
                raise Exception('ML algorithum produced un-recognized data')
            i += 1
        return generated_test, perfect_test,clf

# go throught FB, find the good posts
class Scavange(Classify):
    # alg: 'text', 'expr', 'nomachine'.  'nomachine' is by far the best
    def __init__(self,alg):
        Catigorize.__init__(self)
        Classify.__init__(self)
        self.alg = alg
        if alg == 'expr':
            generated_test, perfect_test, self.clf = self.createMLrows(split=1,method='expr')
        elif alg == 'text':
            generated_test, perfect_test, self.clf = self.createMLrows(split=1, method='expr')
        else:
            self.clf = None

    # messages number from twilio to users phone, with provided message
    def message(self,message):
        account_sid = _account_sid
        auth_token = _auth_token
        client = Client(account_sid, auth_token)
        m = client.messages.create(
            to=_to_phone_number,
            from_=_from_phone_number,
            body=message
        )
        return m.sid

    # fbErrorHandler(e) takes in an exception code e, and handles it.  determines if it is serious or not,
    # if not returns 5 and sleeps for a bit.  Otherwise stops everything.
    def fbErrorHandler(self,e):
        try:
            e.type
        except:
            # sleep for 15 then resume
            sleep(15)
            return 0
            # we beak out of the function early but return 0 because we dont want to break program execution
            # the "ConnectionError" is actually not a facebook error but a network error so it doesnt have a type
        if e.type == "OAuthException" or e.type == "102" or e.type == 102:
            print(e.type)
            print(e.message)
            if not (e.message == "An unknown error has occurred." or e.message == "Error validating access token: This may be because the user logged out or may be due to a system error."):
                self.message(
                    'Dear User,\nThis message is to inform you that your facebook token has expired.\nBest regards,\nyour server'
                )
                return 7
            else:
                # no error, just nework hiccup.  pause and try again.
                sleep(15)
        else:
            # no error, just nework hiccup.  pause and try again.
            time.sleep(15)
        return 5

    # main loop goes onto fb and pulls messages, compares messages, puts new ones in the DB and ignores the rest.
    # if it comes upon a new message for a 4 month selling sublet, it will send a text message.
    def mainloop(self):
        conn = sqlite3.connect('Sublets.db')
        c = conn.cursor()
        start = time.time()
        # since and until are BUGGY.  I just grab the newest 5 messages instead
        since = datetime.datetime.now()
        until = datetime.datetime.now() + datetime.timedelta(days=1)
        try:
            group = self.GroupFeed(num=5,pages=1) # since=since,until=until)
        except Exception as e:
            # couldn't get facebook page messages
            try:
                err = str(e)
            except:
                err = 'Non Identifiable'
            now = str(datetime.datetime.now())
            c.execute('INSERT INTO calls VALUES (?,?,?,?,?,?,?,?)', [now, time.time()-start, time.time()-start, 0, 0, 0, 0, err])
            return self.fbErrorHandler(e)
        fbtime = time.time()
        posts = 0
        newposts = 0
        success = 0
        for data in group['data']:
            try:
                data['message']
            except:
                continue
            posts += 1
            # our well defined constraint
            if self.alg == "nomachine":
                action = self.rank(data['message'])[0]
            elif self.alg == "expr":
                X_test = self.terminology(data['message'], self.period(data['message']) if self.period(data['message']) else 0)
                pred_value = self.clf.predict(X_test.reshape(1,-1))
                action = 'b' if pred_value == 0 else ('s' if pred_value == 1 else None)
            elif self.alg == "text":
                pred_value = self.clf.predict(data['message'])
                action = 'b' if pred_value == 0 else ('s' if pred_value == 1 else None)
            else:
                raise Exception('Unrecognized Algorithum "{}".  Try useing "nomachine", "expr", or "text".'.format(self.alg))
            datarows = [ele[0] for ele in list(c.execute('SELECT * FROM posts'))]
            row = (
                data['created_time'],
                data['from']['name'],
                self.schoolterm(data['created_time']),
                data['message'],
                self.rank(data['message'])[1],
                action,
                self.period(data['message'],priority=[12,8,4]) if action == 's' else self.period(data['message']),
                self.location(data['message']),
                self.price(data['message'])
            )
            if data['created_time'] in datarows:
                continue
            else:
                # put it in, start considering if its important
                newposts += 1
                c.execute('INSERT INTO posts VALUES (?,?,?,?,?,?,?,?,?)', [
                    data['created_time'],
                    data['from']['name'],
                    self.schoolterm(data['created_time']),
                    data['message'],
                    self.rank(data['message'])[1],
                    action,
                    self.period(data['message']),
                    self.location(data['message']),
                    self.price(data['message'])
                ])
                # check for 4 month selling posts
                if (self.period(data['message']) == 4
                    and self.rank(data['message'])[0] == 's'
                    and self.location(data['message'])
                    and self.price(data['message'])):
                    success += 1
                    self.message(
                        '\n'.join([
                            'Dear User,',
                            'your server has found a post that matches your houseing requirements.',
                            'Created Time: {}'.format(data['created_time']),
                            'Posted By: {}'.format(data['from']['name']),
                            'Period: {}'.format(self.period(data['message'])),
                            'Indicator: {}'.format('Selling' if self.rank(data['message'])[0] == 's' else 'Buying'),
                            'Link: {}'.format('https://www.facebook.com/' + data['id'])
                            ])
                    )

        # put record of action in DB
        now = str(datetime.datetime.now())
        c.execute('INSERT INTO calls VALUES (?,?,?,?,?,?,?,?)', [now, time.time() - start, fbtime-start, posts, 1, newposts, success, None])
        conn.commit()
        conn.close()
        end = time.time()
        print('{} total seconds elapsed'.format(end-start))
        print('{} fb time fetch'.format(fbtime-start))





c = Classify()
b = Catigorize()
s = Scavange(alg='nomachine')

if __name__ == '__main__':
    start = datetime.datetime.now()
    end = start + datetime.timedelta(hours=1)
    while end > datetime.datetime.now():
        s.mainloop()
        time.sleep(5)


    # b.getdata(3000)
    # c.tocsv()


    # nomachine = c.loadnomachine()
    # perfect = c.loadperfect()
    # machine, perfect_machine, alg = c.createMLrows(split=.85, method='expr')
    # c.graphTest(nomachine,perfect)
    # c.graphTest(machine,perfect_machine)

    # s = Scavange(alg)
    # s.mainloop()




#.0075 per text
# $1 per month for the number
# Your Account SID from twilio.com/console
