import string

import pandas as pd
data = pd.read_csv("data/news.csv")


def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str


def check(key):
    location = data['text'].loc[data['text'] == key].index.values
    print(location)
    if len(location) == 0:
        print("executed")
        return 0

    else:
        return 1
