import os
import json
import pandas as pd
import librosa
import re
from collections import Counter

def time_to_seconds(time_str):
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", time_str)
    if not match:
        raise ValueError(f"Wrong format: {time_str}")

    hours, minutes, seconds, milliseconds = map(int, match.groups())
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return round(total_seconds, 3)

def convert(wav_dir, csv_file, output_json):

    data = {}
    emotion_counter = Counter()
    sentiment_counter = Counter()

    df = pd.read_csv(csv_file)

    wav_files = sorted(f for f in os.listdir(wav_dir))

    for (index, row), wav_filename in zip(df.iterrows(), wav_files):
        emotion = row["Emotion"][0:3].upper()
        sentiment = row["Sentiment"][0:3].upper()
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
            "emotion": emotion,
            "sentiment": sentiment
        }

        emotion_counter[emotion] += 1
        sentiment_counter[sentiment] += 1

        counter = {
            "emotion": emotion_counter,
            "sentiment": sentiment_counter
        }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    with open("jsons/counter-" + output_json, "w", encoding="utf-8") as f:
        json.dump(counter, f, indent=4, ensure_ascii=False)

    print(f"Done! Output: {output_json} and counter-{output_json}")

convert("MELD-wav16k/wav16k_dev", "MELD-labels/dev_sent_emo.csv", "dev.json")