import os
import json
import random
import pandas as pd
import re
import wave
from collections import Counter, defaultdict

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


    with open("jsons/" + output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Done! Output: {output_json}")

#convert("MELD-wav16k/wav16k_dev", "MELD-labels/dev_sent_emo.csv", "dev.json")

def count(wav_json, output_json):
    emotion_counter = Counter()
    sentiment_counter = Counter()

    with open(wav_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        for wav_filename, meta in data.items():
            emotion = meta["emotion"]
            sentiment = meta["sentiment"]
            emotion_counter[emotion] += 1
            sentiment_counter[sentiment] += 1

        counter = {
            "emotion": emotion_counter,
            "sentiment": sentiment_counter
        }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(counter, f, indent=4, ensure_ascii=False)

    print(f"Done! Output: {output_json}")


#count("jsons/test.json", "jsons/counter_test.json")


def downsample(count_json, wav_json, output_json):
    with open(count_json, "r", encoding="utf-8") as f:
        count_data = json.load(f)

    with open(wav_json, "r", encoding="utf-8") as f:
        wav_data = json.load(f)

    min_count = min(count_data["emotion"].values())

    emotion_groups = defaultdict(list)
    for wav_filename, metadata in wav_data.items():
        emotion = metadata["emotion"]
        emotion_groups[emotion].append((wav_filename, metadata))

    downsampled = {}
    for emotion, samples in emotion_groups.items():
        if len(samples) < min_count:
            print(f"[WARNING] '{emotion}' has only {len(samples)} samples, less than min_count {min_count}")
            selected = samples
        else:
            selected = random.sample(samples, min_count)

        for fname, meta in selected:
            downsampled[fname] = meta
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(downsampled, f, indent=4, ensure_ascii=False)

    print(f"Done! Output: {output_json}")


#downsample("jsons/counter_dev.json", "jsons/dev.json", "jsons/ds_dev.json")

def label_predictions(train_json, predictions_json, output_json):
    with open(train_json, "r") as f:
        train_data = json.load(f)

    with open(predictions_json, "r") as f:
        pred_data = json.load(f)

    results = {}

    for wav_file, meta in train_data.items():
        if wav_file in pred_data:
            predicted = pred_data[wav_file]["predicted_class"]
            true_label = meta["emotion"]

            results[wav_file] = {
                "true_label": true_label,
                "predicted_class": predicted
            }

    # Kiírás JSON-be (vagy mehetne CSV-be is)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Done! Output: {output_json}")

label_predictions("jsons/ds_test.json", "results/567422/prediction_outputs.json", "results/567422/labels.json")