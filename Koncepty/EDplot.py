import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib as mpl

problem_count = 4

#with open("computed/2 problems.json") as f:
with open("computed/KNN {0} problems.json".format(problem_count)) as f:
    jLR = json.load(f)
jLR.sort(key=lambda x: -x["E-D"])

with open("computed/{0} problems.json".format(problem_count)) as f:
    jKNN = json.load(f)
jKNN.sort(key=lambda x: -x["E-D"])

with open("computed/model {0} problems.json".format(problem_count)) as f:
    jMOD = json.load(f)
jMOD.sort(key=lambda x: -x["E-D"])

y_pos = np.arange(len(jLR))

maximum_error_rate = 1
bar_height = 0.25
space = 0.05

plt.figure(figsize=(12, 10))
ax = plt.subplot2grid((20, 1), (0, 0), rowspan=17)
plt.title("The most successful approach for given expert error rate")
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('none')
plt.yticks(y_pos, [line["names"] for line in jLR])
plt.xlabel('Error rate of expert')

max_height = max([line["height"] for line in jLR] + [line["height"] for line in jKNN] + [line["height"] for line in jMOD])



p1 = plt.barh(y_pos + bar_height + space, [maximum_error_rate] * len(jLR), bar_height, align='center', alpha=1, color="white", hatch="///")
for y, line in enumerate(jLR):
    p3 = plt.barh(y + bar_height + space, line["ED-D"], bar_height, align='center', color=str(min(1, 1-line["height"]/max_height)))
p2 = plt.barh(y_pos + bar_height + space, [line["E-ED"] for line in jLR], bar_height, align='center', alpha=1, color="white", hatch="xxx")

plt.barh(y_pos + 0, [maximum_error_rate] * len(jKNN), bar_height, align='center', alpha=1, color="white", hatch="///")
for y, line in enumerate(jKNN):
    plt.barh(y + 0, line["ED-D"], bar_height, align='center', color=str(min(1, 1-line["height"]/max_height)))
plt.barh(y_pos + 0, [line["E-ED"] for line in jKNN], bar_height, align='center', alpha=1, color="white", hatch="xxx")

plt.barh(y_pos - bar_height - space, [maximum_error_rate] * len(jMOD), bar_height, align='center', alpha=1, color="white", hatch="///")
for y, line in enumerate(jMOD):
    plt.barh(y - bar_height - space, line["ED-D"], bar_height, align='center', color=str(min(1, 1-line["height"]/max_height)))
plt.barh(y_pos - bar_height - space, [line["E-ED"] for line in jMOD], bar_height, align='center', alpha=1, color="white", hatch="xxx")

for y, line in enumerate(jLR):
    #plt.text((line["E-ED"] + line["ED-D"]) / 2, y, "{0:.2%}".format(line["height"]), horizontalalignment="center",)
    p4 = plt.plot([line["E-D"], line["E-D"]], [y+bar_height*1.5 + space, y-bar_height*1.5 - space], "k:")
    plt.plot([line["E-D"], line["E-D"]], [y + bar_height*1.5 + space - 0.1, y - bar_height*1.5 - space], "w:")

plt.legend( (p1[0], p2[0], p3[0], p4[0]), ('Data', 'Expert', 'Correction of expert', 'Data = Expert') )


ax2 = ax.twinx()
y_pos2 = []
for i, line in enumerate(jLR):
    y_pos2 += [i-bar_height-space, i, i+bar_height+space]
ax2.set_yticks(y_pos2)
ax2.set_yticklabels(["model", "logReg", "KNN"] * len(jLR))
ax2.yaxis.set_ticks_position('none')


ax1 = plt.subplot2grid((20, 1), (19, 0))
a = np.array([[line["height"] for line in jLR] + [line["height"] for line in jKNN] + [line["height"] for line in jMOD]])
a = np.maximum(a, np.array([0]*len(a)))
a = np.max(a)

cb = mpl.colorbar.ColorbarBase(ax1, cmap=mpl.cm.binary, orientation='horizontal')
ticks = np.arange(11) / 10.
cb.set_ticks(ticks)
cb.set_ticklabels(["{0:.1%}".format(x) for x in ticks * a])
cb.set_label('Maximal improvement over expert and data')

plt.show()