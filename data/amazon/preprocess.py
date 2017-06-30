
# coding: utf-8

# prepare amazon data as an input to to cuMF
# data from http://jmcauley.ucsd.edu/data/amazon/
# each line is like "user_id item_id rating timestamp"
import os
from six.moves import urllib
import numpy as np
import csv
import random
import pandas as pd
import math
import matplotlib.pyplot as plt

np.random.seed(0)

url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/'
# datafile = 'ratings_Books' # 43201
# datafile = 'ratings_Electronics' # 520
datafile = 'ratings_Movies_and_TV' # 43935
# datafile = 'ratings_top3' # 

csvfile =  datafile + '.csv'
trainfile = datafile + '_train'
testfile = datafile + '_test'


names = ['user_id', 'item_id', 'rating', 'timestamp']

if not os.path.exists(datafile):
    if not os.path.exists(csvfile):
        urllib.request.urlretrieve(url + csvfile, csvfile)
    users = {}
    items = {}
    user_cnt = 0
    item_cnt = 0
    df = pd.read_csv(csvfile, sep=',', names=names)
    with open(datafile, "wb") as outfile:
        writer = csv.writer(outfile, delimiter=' ')
        for row in df.itertuples():
            if row[1] in users:
                user_id = users[row[1]]
            else:
                users[row[1]] = user_cnt
                user_id = user_cnt
                user_cnt += 1
            if row[2] in items:
                item_id = items[row[2]]
            else:
                items[row[2]] = item_cnt
                item_id = item_cnt
                item_cnt += 1
            writer.writerow([user_id, item_id, row[3]])

names = ['user_id', 'item_id', 'rating']
df = pd.read_csv(datafile, sep=' ', names=names)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
n_ratings = df.shape[0]
print str(n_users) + ' users'
print str(n_items) + ' items'
print str(n_ratings) + ' ratings'
sparsity = float(n_ratings)/(n_items * n_users)
sparsity *= 100
print 'Sparsity: {:4.2f}%'.format(sparsity)

# ur, b = np.histogram(df.user_id.values, bins=n_users, density=False)
# print max(ur)
# hist, bins = np.histogram(ur, bins=np.arange(1, 25), density=True)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.title(datafile)
# plt.xlabel('number of ratings')
# plt.ylabel('user distribution')
# plt.savefig(datafile+'.png')

# # limited by memory
# ratings = np.zeros((n_users, n_items))
# for row in df.itertuples():
#     ratings[row[1], row[2]] = row[3]


# def train_test_split(ratings):
#     test = np.zeros(ratings.shape)
#     train = ratings.copy()
#     for user in xrange(ratings.shape[0]):
#         cnt = len(ratings[user, :].nonzero()[0])
#         if cnt > 100:
#             test_ratings = np.random.choice(ratings[user, :].nonzero()[0], size=(cnt-1)/100+1, replace=False)
#             train[user, test_ratings] = 0.
#             test[user, test_ratings] = ratings[user, test_ratings]
#     # Test and training are truly disjoint
#     assert(np.all((train * test) == 0)) 
#     return train, test

# def write_to_file(m, file):
#     (rows, cols) = np.where(m)
#     vals = m[rows, cols]
#     tri = zip(rows, cols, vals)
#     with open(file, "wb") as outfile:
#         writer = csv.writer(outfile, delimiter=' ')
#         for row in tri:
#             writer.writerow(row)

# train, test = train_test_split(ratings)
# write_to_file(train, trainfile)
# write_to_file(test, testfile)
