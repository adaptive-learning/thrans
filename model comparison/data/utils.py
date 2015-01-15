from collections import defaultdict
import json
from hashlib import sha1
import numpy as np
import os
import pandas as pd
from proso.geography import places, answers
from data import Data


def compute_correlations(data, method="spearman", guess_decay=True, test=True,  min_periods=1, hits=False):
    if test:
        df = data.get_train_dataframe()
    else:
        df = data.get_dataframe_all()

    filename = "data/{}.respo.pd".format(sha1(str(data) + str(test) + str(guess_decay) + "v2").hexdigest()[:10])
    try:
        responses = pd.read_pickle(filename)
        print "Loaded response matrix"
    except:
        print "Computing response matrix"
        responses = pd.DataFrame(index=df["student"].unique(), columns=sorted(df["item"].unique()))
        for answer in data.all_iter() if not test else data.train_iter():
            guess = 0
            if guess_decay and answer["choices"]:
                guess = 1. / answer["choices"]
            responses.ix[answer["student"], answer["item"]] = answer["correct"] * 1 - guess
        responses.to_pickle(filename)

    if hits:
        hits = pd.DataFrame(columns=responses.columns, index=responses.columns)
        print "Computing hits"
        mask = pd.isnull(responses)
        for p1 in responses.columns:
            for p2 in responses.columns:
                valid = 1 - (mask[p1] | mask[p2])
                hits.ix[p1, p2] = valid.sum()

    print "Computing correlations"
    responses = responses.astype(float)
    corr = responses.corr(method=method, min_periods=min_periods)
    corr.fillna(0, inplace=True)
    print "NaNs in correlation matrix", (corr==0).sum().sum()

    return corr, hits, (corr==0).sum().sum()

def hash(model, data):
    return sha1(str(model)+str(data)).hexdigest()[:10]


def old_geography_data_parser(output, input="raw data/data-geography-first.csv"):
    with open(input) as fin, open(output, "w") as fout:
        fin.readline()

        for line in fin.readlines():
            line = line.split(",")
            qtype = int(line[4])
            answer = {
                "student": int(line[0]),
                "item": int(line[1]),
                "correct": line[3] == "1",
                "extra": {
                    "answer": int(line[2]),
                    "choices": 0 if qtype == 10 else (qtype - 10 * (qtype//10)),
                    "time": int(line[5]),
                },
            }
            fout.write(json.dumps(answer)+"\n")


def geography_data_parser(output, input="raw data/geography-all.csv"):
    data = pd.DataFrame.from_csv(input, index_col=False)
    data["correct"] = data["place_asked"] == data["place_answered"]
    data.rename(inplace=True, columns={
        "user": "student",
        "place_asked": "item",
        "place_answered": "answer",
        "response_time": "time",
        "number_of_options": "choices",
        "place_map": "map"
    })
    del data["options"]
    del data["inserted"]

    data["index"] = data["id"]
    data.set_index("index", inplace=True)
    data.save(output)


def filter_first(input="geography-all-2.pd", output=None):
    data = pd.load(input)

    data.sort("id", inplace=True)

    filtered = data.drop_duplicates(['student', 'item'])
    filtered["index"] = filtered["id"]
    filtered.set_index("index", inplace=True)

    filtered.save(output)
    print filtered


def filter_states(input="geography-first-all-2.pd", output=None):
    data = pd.load(input)

    places = reduce(list.__add__, get_continents_country_maps().values(), [])

    filtered = data[data["item"].isin(places)]

    filtered.save(output)
    print filtered


def filter_europe(input="geography-first-all-2.pd", output=None):
    data = pd.load(input)

    places = get_maps()["Europe-country"]
    filtered = data[data["item"].isin(places)]

    filtered.save(output)
    print filtered


def filter_cz_cities(input="geography-first-all-2.pd", output=None):
    data = pd.load(input)

    places = get_maps()["Czech Rep.-city"]
    filtered = data[data["item"].isin(places)]

    filtered.save(output)
    print filtered


def filter_small_data(input="geography-first-all.pd-2", output=None, min_students=200, min_items=20):
    data = pd.load(input)

    valid_users = map(
        lambda (u, n): u,
        filter(
            lambda (u, n): n >= min_items,
            data.groupby('student').apply(len).to_dict().items()
        )
    )
    data = data[data["student"].isin(valid_users)]

    valid_items = map(
        lambda (u, n): u,
        filter(
            lambda (u, n): n >= min_students,
            data.groupby('item').apply(len).to_dict().items()
        )
    )
    data = data[data["item"].isin(valid_items)]

    data.save(output)
    print data



def find_maps(just_types=False):
    types = {
        1: "country",
        2: "city",
        3: "world",
        4: "continent",
        5: "river",
        6: "lake",
        7: "region-(cz)",
        8: "bundesland",
        9: "province",
        10: "region-(it)",
        11: "region",
        12: "autonomus-community",
        13: "mountains",
        14: "island",
    }

    # answers_all = answers.from_csv("raw data/geography-all.csv")
    maps = defaultdict(lambda: [])
    places_all = places.from_csv("raw data/geography.place.csv", "raw data/geography.placerelation.csv","raw data/geography.placerelation_related_places.csv")
    places_all.set_index(places_all["id"], inplace=True)
    for _, place in places_all.iterrows():
        for id, relation in place.relations:
            if relation == "is_on_map":
                if just_types:
                    key = types[place.type]
                else:
                    key = places_all.ix[id]["name"]+"-"+types[place.type]
                maps[key].append(place.id)

    filename = "types.json" if just_types else "maps.json"
    with open(filename, "w") as f:
        json.dump(maps, f)


def get_maps(folder="", filter=None, just_types=False):
    filename = "types.json" if just_types else "maps.json"
    with open(folder+filename) as f:
        maps = json.load(f)

    if filter is not None:
        new = {}
        for map in filter:
            new[map] = maps[map]
        maps = new

    return maps


def get_id_place_map(dir=""):
    places = pd.read_csv(dir+"raw data/geography.place.csv", index_col=False)
    map = {}
    for _, place in places.iterrows():
        map[place.ix["id"]] = place.ix["name_en"]
    return map


def get_continents_country_maps(folder=""):
    return get_maps(folder, ["United States-country", "Australia-country", u"Jizni Amerika-country", "Africa-country", "Asia-country", u"Severni Amerika-country", "Europe-country"])


def get_train_students():
    train = pd.DataFrame.from_csv("raw data/answers_train.csv", index_col=False)
    # test = pd.DataFrame.from_csv("raw data/answers_test.csv", index_col=False)
    # all = pd.DataFrame.from_csv("raw data/answers_all.csv", index_col=False)

    # print len(set(train["user"]))
    # print len(set(test["user"]))
    # print len(set(all["user"]))
    json.dump(list(train["user"]), open("train_users.json", "w"))

europe_clusters = {'europe-1': [51, 66, 70, 74, 78, 147, 154, 190, 196, 234], 'europe-0': [79, 88, 108, 113, 114, 115, 144, 146, 159, 176, 178, 179, 182, 184, 191, 194, 203, 216], 'europe-2': [64, 81, 93, 94, 142, 143, 164, 165, 181, 205, 206]}

# maps = get_maps()
# places_all = places.from_csv("raw data/geography.place.csv", "raw data/geography.placerelation.csv","raw data/geography.placerelation_related_places.csv")
# places_all.set_index(places_all["id"], inplace=True)
# print places_all.ix[482]
# for p in maps["Czech Rep.-city"]:
#     print places_all.ix[p]



# geography_data_parser("geography-all-2.pd", input="raw data/answers_all.csv")
# filter_first(output="geography-first-all-2.pd")
# filter_states(output="geography-first-states-2.pd")
# filter_europe(output="geography-first-europe-2.pd")
# filter_cz_cities(output="geography-first-cz_city-2.pd")
# filter_small_data(input="geography-first-states-2.pd", output="geography-first-states-filtered-2.pd", min_items=10)
# filter_small_data(input="geography-first-cz_city-2.pd", output="geography-first-cz_city-filtered-2.pd", min_items=10)
# filter_small_data(input="geography-first-europe-2.pd", output="geography-first-europe-filtered-2.pd", min_items=10)

# filter_small_data(input="geography-first-all-2.pd", output="geography-first-all-filtered-2.pd", min_items=10)
# d = Data("geography-first-all.pd", train=0.3)
# d.get_train_dataframe().to_csv("train-all.csv")

# get_train_students()

# find_maps(just_types=True)