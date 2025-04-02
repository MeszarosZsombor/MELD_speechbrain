import sys

from speechbrain.utils.metric_stats import ClassificationStats

cs = ClassificationStats()

cs.append(

    ids=["ITEM1", "ITEM2", "ITEM3", "ITEM4"],

    predictions=[
        "JOY",
        "SAD",
        "NEU",
        "ANG"
    ],
    targets=[
        "JOY",
        "SAD",
        "DIS",
        "ANG"
    ],

    categories=["SAD", "SUR", "NEU", "JOY", "ANG", "DIS", "FEA"]

)

cs.write_stats(sys.stdout)

summary = cs.summarize()

print(summary['accuracy'])
