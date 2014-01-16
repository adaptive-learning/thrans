import matplotlib.pyplot as plt
import json
import numpy as np

#with open("computed/2 problems.json") as f:
with open("computed/KNN 5 problems.json") as f:
    j = json.load(f)

for line in j:
    pass

j.sort(key=lambda x: -x["E-D"])

maximum_error_rate = 1
y_pos = np.arange(len(j))

plt.barh(y_pos, [maximum_error_rate] * len(j), align='center', alpha=1, color="green")
plt.barh(y_pos, [line["ED-D"] for line in j], align='center', alpha=1, color="red")
plt.barh(y_pos, [line["E-ED"] for line in j], align='center', alpha=1, color="blue")
plt.yticks(y_pos, [line["names"] for line in j])


for y, line in enumerate(j):
    plt.text((line["E-ED"] + line["ED-D"]) / 2, y, "{0:.2%}".format(line["height"]), horizontalalignment="center",)

    plt.plot([line["E-D"], line["E-D"]], [y+0.4, y-0.4], "w:")

plt.xlabel('Error rate of expert')

plt.show()

