import os
import json
import pandas as pd
import librosa
import re

def time_to_seconds(time_str):
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", time_str)
    if not match:
        raise ValueError(f"Wrong format: {time_str}")

    hours, minutes, seconds, milliseconds = map(int, match.groups())
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return round(total_seconds, 3)

wav_dir = "MELD-wav16k/wav16k_train"
csv_file = "MELD-labels/train_sent_emo.csv"
output_json = "train.json"

data = {}

df = pd.read_csv(csv_file)

df = df.head(20)

wav_files = sorted(f for f in os.listdir(wav_dir))

for (index, row), wav_filename in zip(df.iterrows(), wav_files):
    label = row["Emotion"][0].upper() + row["Sentiment"][0].upper()
    start = row.get("StartTime", None)
    end = row.get("EndTime", None)

    wav_path = os.path.join(wav_dir, wav_filename)

    if pd.notna(start) and pd.notna(end):
        length = time_to_seconds(end) - time_to_seconds(start)
    else:
        duration = librosa.get_duration(path=wav_path)
        length = round(duration)

    data[wav_filename] = {
        "wav": "{data_root}/" + wav_path,
        "length": length,
        "label": label
    }

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Done! Output: {output_json}")