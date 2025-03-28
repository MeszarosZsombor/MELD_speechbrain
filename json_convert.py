import os
import json
import pandas as pd
import librosa
import re
import wave
from collections import Counter

def time_to_seconds(time_str):
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", time_str)
    if not match:
        raise ValueError(f"Wrong format: {time_str}")

    hours, minutes, seconds, milliseconds = map(int, match.groups())
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return round(total_seconds, 3)

def get_wav_length(wav_path):
    """Kiolvassa a WAV fájl hosszát másodpercekben."""
    with wave.open(wav_path, "r") as wf:
        frames = wf.getnframes()  # Összes minta
        rate = wf.getframerate()  # Mintavételezési gyakoriság
        duration = frames / float(rate)  # Másodpercekben
        return round(duration, 3)  # Három tizedesjegyre kerekítve

def convert(wav_dir, csv_file, output_json):

    data = {}
    emotion_counter = Counter()
    sentiment_counter = Counter()

    df = pd.read_csv(csv_file)

    wav_files = sorted(f for f in os.listdir(wav_dir))

    for (index, row), wav_filename in zip(df.iterrows(), wav_files):
        emotion = row["Emotion"][0:3].upper()
        sentiment = row["Sentiment"][0:3].upper()

        wav_path = os.path.join(wav_dir, wav_filename)

        length = get_wav_length(wav_path)

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

    with open("jsons/" + output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    with open("jsons/counter-" + output_json, "w", encoding="utf-8") as f:
        json.dump(counter, f, indent=4, ensure_ascii=False)

    print(f"Done! Output: {output_json} and counter-{output_json}")

convert("MELD-wav16k/wav16k_dev", "MELD-labels/dev_sent_emo.csv", "dev.json")