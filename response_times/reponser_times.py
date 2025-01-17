import json
import os
import re
from time import time
from datetime import timedelta
from proso.geography import answers as ga
import pandas as pd
import pylab as plt
import seaborn as sns
import numpy as np
from scipy import stats


def get_answers(from_csv=False, sample=False, min_answers_per_item=0, min_answers_per_user=10):
    if from_csv or not os.path.exists("data/answers{}.pd".format(".sample" if sample else "")):
        answers = ga.from_csv("data/geography.answer{}.csv".format(".sample" if sample else ""))
        answers = answers[answers["response_time"] < np.percentile(answers["response_time"], 99)]
        answers = answers[answers["response_time"] > 0]
        answers["correct"] = answers["place_asked"] == answers["place_answered"]
        answers = answers[answers.join(pd.Series(answers.groupby("place_asked").apply(len), name="count"), on="place_asked")["count"] > min_answers_per_item]
        answers = answers[answers.join(pd.Series(answers.groupby("user").apply(len), name="count"), on="user")["count"] > min_answers_per_user]
        answers["log_times"] = np.log(answers["response_time"] / 1000)
        answers = answers.join(np.exp(pd.Series(answers.groupby("user")["log_times"].apply(np.mean), name="user_mean")), on="user")
        answers = answers.join(pd.Series(answers.groupby("user")["response_time"].apply(np.std), name="user_std"), on="user")
        answers = answers.join(pd.Series(answers.groupby("user")["correct"].apply(np.mean), name="user_sr"), on="user")
        answers = answers.join(pd.Series(answers.groupby("user")["log_times"].apply(np.std), name="user_std_log"), on="user")
        answers = join_feedback(answers)
        answers = join_difficulties(answers)
        answers = join_skills(answers)
        answers = answers.join(pd.Series(answers.groupby("user")["inserted"].apply(np.max) - answers.groupby("user")["inserted"].apply(np.min) > timedelta(days=1), name="returning"), on="user")

        answers .to_pickle("data/answers{}.pd".format(".sample" if sample else ""))
    else:
        answers = pd.read_pickle("data/answers{}.pd".format(".sample" if sample else ""))

    return answers


def mark_class_room_users(answers, classroom_size=20):
    classroom_users = [
        user
        for ip, users in (
            answers.sort('id').drop_duplicates('user').
            groupby('ip_address').
            apply(lambda x: x['user'].unique()).
            to_dict().
            items())
        for user in users
        if len(users) > classroom_size
    ]
    answers["class"] = False
    answers.loc[answers["user"].isin(classroom_users), "class"] = True

def split_by_mean_time(answers, count=11, size=1, start=1):
    answers["speed"] = "longer"

    for bin in range(count):
        time = bin * size + start
        answers.loc[(time < answers["user_mean"]) & (answers["user_mean"] < (time + size)), "speed"] = "{}s-{}s".format(time, time + size)


def join_feedback(answers):
    feedback = pd.read_csv("data/feedback.rating.csv")
    answers = answers.join(pd.Series(feedback.groupby("user").apply(lambda x: x["value"].mean()), name="feedback"), on="user")
    answers = answers.join(pd.Series(feedback.groupby("user").first()["value"], name="first_feedback"), on="user")
    return answers


def join_difficulties(answers):
    diffs = pd.read_csv("data/difficulties.csv").set_index("place")
    answers = answers.join(diffs, on="place_asked")
    return answers

def join_skills(answers):
    skills = pd.read_csv("data/skills.csv").set_index("user")
    answers = answers.join(skills, on="user")
    answers["skill"].fillna(0, inplace=True)
    return answers


def log_mean_time_hist(answers, classes=False):
    plt.title("Histogram of log-meanswers of response time of user")

    if classes:
        answers[~answers["class"]].groupby("user").first()["user_mean"].hist(bins=50, range=(0, 15), label="other")
        answers[answers["class"]].groupby("user").first()["user_mean"].hist(bins=50, range=(0, 15), label="in class")
        plt.legend(loc=1)
    else:
        answers.groupby("user").first()["user_mean"].hist(bins=50, range=(0, 15))
    # plt.xlim(0, 30)


def join_concepts(answers, concepts):
    concepts = json.load(open(concepts))
    answers["concept"] = "unknown"
    for name, places in concepts.items():
        answers.loc[answers["place_asked"].isin(places), "concept"] = name


def timesort(value):
    val = re.match("^(\d+)", value)
    val = int(val.group(0)) if val else 999
    return val


def compare_speed_and_accuracy(answers):
    speeds = sorted(answers["speed"].unique(), key=timesort)
    srs = []
    counts = []
    for speed in speeds:
        sr = answers[answers["speed"] == speed].groupby("user")["correct"].mean().mean()
        srs.append(sr)
        counts.append(len(answers.loc[answers["speed"] == speed]["user"].unique()))

    plt.figure()
    ax1 = plt.subplot()
    ax1.plot(range(len(speeds)), srs)
    plt.xticks(range(len(speeds)), speeds)
    ax1.set_xlabel("log-mean time")
    ax1.set_ylabel("avg success rate")
    ax2 = ax1.twinx()
    ax2.plot(range(len(speeds)), counts, "g")
    ax2.set_ylabel("user count")


def compare_speed_and_feedback(answers):
    speeds = sorted(answers["speed"].unique(), key=timesort)
    feedbacks = []
    counts = []
    for speed in speeds:
        feedback = answers[answers["speed"] == speed].groupby("user")["feedback"].mean().mean()
        feedbacks.append(feedback)
        counts.append(len(answers.loc[(answers["speed"] == speed) & pd.notnull(answers["feedback"])]["user"].unique()))

    plt.figure()
    ax1 = plt.subplot()
    ax1.plot(range(len(speeds)), feedbacks)
    plt.xticks(range(len(speeds)), speeds)
    ax1.set_xlabel("log-mean time")
    ax1.set_ylabel("avg of avg feedback")
    ax2 = ax1.twinx()
    ax2.plot(range(len(speeds)), counts, "g")
    ax2.set_ylabel("user count")


def compare_speed_and_difficulty(answers):
    speeds = sorted(answers["speed"].unique(), key=timesort)
    diffs = []
    counts = []
    for speed in speeds:
        diff = answers[answers["speed"] == speed].groupby("user")["difficulty"].mean().mean()
        diffs.append(diff)
        counts.append(len(answers.loc[(answers["speed"] == speed)]["user"].unique()))

    plt.figure()
    ax1 = plt.subplot()
    ax1.plot(range(len(speeds)), diffs)
    plt.xticks(range(len(speeds)), speeds)
    ax1.set_xlabel("log-mean time")
    ax1.set_ylabel("mean of avg difficulty")
    ax2 = ax1.twinx()
    ax2.plot(range(len(speeds)), counts, "g")
    ax2.set_ylabel("user count")

def compare_speed_and_skill(answers):
    speeds = sorted(answers["speed"].unique(), key=timesort)
    skills = []
    counts = []
    for speed in speeds:
        skill = answers[answers["speed"] == speed].groupby("user")["skill"].first().mean()
        skills.append(skill)
        counts.append(len(answers.loc[(answers["speed"] == speed)]["user"].unique()))

    plt.figure()
    ax1 = plt.subplot()
    ax1.plot(range(len(speeds)), skills)
    plt.xticks(range(len(speeds)), speeds)
    ax1.set_xlabel("log-mean time")
    ax1.set_ylabel("mean of skill")
    ax2 = ax1.twinx()
    ax2.plot(range(len(speeds)), counts, "g")
    ax2.set_ylabel("user count")


def compare_speed_and_returning(answers):
    speeds = sorted(answers["speed"].unique(), key=timesort)
    skills = []
    counts = []
    for speed in speeds:
        skill = answers[answers["speed"] == speed].groupby("user")["returning"].first().mean()
        skills.append(skill)
        counts.append(len(answers.loc[(answers["speed"] == speed)]["user"].unique()))

    plt.figure()
    ax1 = plt.subplot()
    ax1.plot(range(len(speeds)), skills)
    plt.xticks(range(len(speeds)), speeds)
    ax1.set_xlabel("log-mean time")
    ax1.set_ylabel("returning rate")
    ax2 = ax1.twinx()
    ax2.plot(range(len(speeds)), counts, "g")
    ax2.set_ylabel("user count")

def compare_speed_and_class(answers):
    speeds = sorted(answers["speed"].unique(), key=timesort)
    class_rates = []
    counts = []
    for speed in speeds:
        class_rate = answers[answers["speed"] == speed]["class"].mean() * 100
        class_rates.append(class_rate)
        counts.append(len(answers.loc[(answers["speed"] == speed)]["user"].unique()))

    plt.figure()
    ax1 = plt.subplot()
    ax1.plot(range(len(speeds)), class_rates)
    plt.xticks(range(len(speeds)), speeds)
    ax1.set_xlabel("log-mean time")
    ax1.set_ylabel("% users from class")
    ax2 = ax1.twinx()
    ax2.plot(range(len(speeds)), counts, "g")
    ax2.set_ylabel("user count")


def compare_speed_and_answers(answers, time=False):
    speeds = sorted(answers["speed"].unique(), key=timesort)
    if time:
        speeds = speeds[:-1]
    anss = []
    counts = []
    for speed in speeds:
        ans = answers[answers["speed"] == speed].groupby("user").apply(len).mean()
        t = answers[answers["speed"] == speed].groupby("user")["user_mean"].first().mean() if time else 1
        anss.append(ans * t)
        counts.append(len(answers.loc[(answers["speed"] == speed)]["user"].unique()))

    plt.figure()
    ax1 = plt.subplot()
    ax1.plot(range(len(speeds)), anss)
    plt.xticks(range(len(speeds)), speeds)
    ax1.set_xlabel("log-mean time")
    ax1.set_ylabel("avg time spend in system" if time else "avg of number of answers")
    ax2 = ax1.twinx()
    ax2.plot(range(len(speeds)), counts, "g")
    ax2.set_ylabel("user count")


def rs_by_concept(answers, concepts):
    join_concepts(answers, concepts)
    concepts = answers["concept"].unique()
    data = []
    for concept in concepts:
        data.append(answers[answers["concept"] == concept]["response_time"].median() / 1000)

    data, concepts = zip(*sorted(zip(data, concepts), key=lambda x: -x[0]))

    plt.bar(range(len(concepts)), data)
    plt.xticks(np.arange(len(concepts)) + 0.5, concepts, rotation=80)
    plt.ylabel("median of response time")


def rs_distribution(answers, split="speed"):
    order = sorted(answers["speed"].unique(), key=timesort) if split == "speed" else None
    sns.violinplot(answers["log_times"], answers[split], order=order)
    plt.xlabel(split)


def std_distribution(answers, split="speed"):
    order = sorted(answers["speed"].unique(), key=timesort) if split == "speed" else None
    answers = answers.groupby("user").first()
    sns.violinplot(answers["user_std_log"], answers[split], order=order)
    plt.xlabel(split)


def compare_speed_and_std(answers):
    speeds = sorted(answers["speed"].unique(), key=timesort)
    class_rates = []
    counts = []
    for speed in speeds:
        class_rate = answers[answers["speed"] == speed].groupby("user")["user_std_log"].mean().mean()
        class_rates.append(class_rate)
        counts.append(len(answers.loc[(answers["speed"] == speed)]["user"].unique()))

    plt.figure()
    ax1 = plt.subplot()
    ax1.plot(range(len(speeds)), class_rates)
    plt.xticks(range(len(speeds)), speeds)
    ax1.set_xlabel("log-mean time")
    ax1.set_ylabel("avg of std of users log_times")


def compare_sr_and_speed(answers):
    speeds = []
    counts = []
    start = 30
    stop = 100
    step = 5
    for sr in np.arange(start, stop, step) / 100.:
        speed = answers[(answers["user_sr"] > sr) & ((sr + step / 100.) > answers["user_sr"])].groupby("user")["user_mean"].mean().mean()
        speeds.append(speed)
        counts.append(len(answers.loc[(answers["user_sr"] > sr) & ((sr + step / 100.) > answers["user_sr"])]["user"].unique()))

    plt.figure()
    ax1 = plt.subplot()
    ax1.plot(range(start, stop, step), speeds)
    plt.xticks(range(start, stop, step), map(lambda x: "{}-{}%".format(x, x+ step), range(start, stop, step)))
    ax1.set_xlabel("success rate")
    ax1.set_ylabel("avg of log-mean time per user")
    ax2 = ax1.twinx()
    ax2.plot(range(start, stop, step), counts, "g")
    ax2.set_ylabel("user count")


def compare_diff_and_speed(answers):
    speeds = []
    counts = []
    start = -3.5
    stop = 5
    step = 0.5
    for diff in np.arange(start, stop, step):
        speed = np.exp(answers[(answers["difficulty"] > diff) & ((diff + step) > answers["difficulty"])]["log_times"].mean())
        speeds.append(speed)
        counts.append(len(answers.loc[(answers["difficulty"] > diff) & ((diff + step) > answers["difficulty"])]))

    plt.figure()
    ax1 = plt.subplot()
    ax1.plot(np.arange(start, stop, step), speeds)
    plt.xticks(np.arange(start, stop, step), map(lambda x: "{}-{}".format(x, x+ step), np.arange(start, stop, step)))
    ax1.set_xlabel("difficulty")
    ax1.set_ylabel("avg of log-mean times")
    ax2 = ax1.twinx()
    ax2.plot(np.arange(start, stop, step), counts, "g")
    ax2.set_ylabel("answers")


def report(answers):
    print "answers", len(answers)
    print "facts", len(answers["place_asked"].unique())
    print "users", len(answers["user"].unique())
    print "feedback", len(pd.read_csv("data/feedback.rating.csv"))
    print "user with feedback", len(pd.read_csv("data/feedback.rating.csv")["user"].unique())


answers = get_answers(sample=True, from_csv=True)
# mark_class_room_users(answers)
split_by_mean_time(answers)

# compare_speed_and_accuracy(answers)
# compare_speed_and_feedback(answers)
# compare_speed_and_difficulty(answers)
compare_speed_and_skill(answers)
compare_speed_and_returning(answers)
# compare_speed_and_class(answers)
# compare_speed_and_answers(answers)
# compare_speed_and_answers(answers, time=True)
# compare_speed_and_std(answers)
# rs_by_concept(answers, "data/maps-types.json")
# rs_distribution(answers)
# rs_distribution(answers, "class")
# rs_distribution(answers, "first_feedback")

# compare_diff_and_speed(answers)

# answers["difficulty_groups"] = answers["difficulty"].apply(round)
# rs_distribution(answers)
# std_distribution(answers)
# std_distribution(answers, "class")
# std_distribution(answers, "first_feedback")

# log_mean_time_hist(answers, classes=True)


# print stats.ttest_ind(answers[answers["class"]].groupby("user")["user_std"].mean(), answers[~answers["class"]].groupby("user")["user_std"].mean())
# report(answers)

# t = time()
# print len(answers.groupby(lambda x: answers.ix[x, "inserted"].year)["response_time"].mean())
# answers.set_index("inserted", inplace=True)
# print time() - t
# print len(answers.groupby(lambda t: t.year)["response_time"].mean())
# print time() - t
plt.show()