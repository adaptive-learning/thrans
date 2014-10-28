import json
from hashlib import sha1

def hash(model, data):
    return sha1(str(model)+str(data)).hexdigest()[:10]

def geography_data_parser(output, input="raw data/data-geography-first.csv"):
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

# geography_data_parser("geography-first-all.json")