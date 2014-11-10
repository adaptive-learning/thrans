import json
from hashlib import sha1
import pandas as pd


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


def filter_first(input="geography-all.pd", output=None):
    data = pd.load(input)

    filtered = data.groupby(["student"]). \
        apply(lambda x: x.drop_duplicates('item'))
    filtered["index"] = filtered["id"]
    filtered.set_index("index", inplace=True)

    filtered.save(output)
    print filtered


def filter_states(input="geography-first-all.pd", output=None):
    data = pd.load(input)

    filtered = data[data["type"] == 1]

    filtered.save(output)
    print filtered


def filter_europe(input="geography-first-all.pd", output=None):
    data = pd.load(input)

    filtered = data[data["item"].isin([48, 58, 60, 64, 68, 72, 73, 75, 82, 87, 88, 102, 107, 108, 109, 136, 137, 138, 140, 141, 148, 153, 158, 159, 170, 172, 173, 175, 176, 178, 184, 185, 188, 190, 197, 199, 200, 210, 226])]

    filtered.save(output)
    print filtered


def filter_small_data(input="geography-first-all.pd", output=None, min_students=200, min_items=20):
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


# geography_data_parser("geography-all.pd")
# filter_first(output="geography-first-all.pd")
# filter_states(output="geography-first-states.pd")
# filter_europe(output="geography-first-europe.pd")
# filter_small_data(input="geography-first-europe.pd", output="geography-first-europe-filtered.pd", min_items=10)