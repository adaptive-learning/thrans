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

    data.save(output)


def filter_first(input="geography-all.pd", output=None):
    data = pd.load(input)

    filtered = data.groupby(["student"]). \
        apply(lambda x: x.drop_duplicates('item')). \
        set_index("id").\
        reset_index()
    filtered.save(output)

    print filtered


# geography_data_parser("geography-all.pd")
filter_first(output="geography-first-all.pd")