# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 11:25:58 2021

@author: xiatong
"""

import joblib
import pandas as pd

import data_params as params


user_all = {
    "train_asthma_id": [],
    "vad_asthma_id": [],
    "test_asthma_id": [],
    "train_health_id": [],
    "vad_health_id": [],
    "test_health_id": [],
}

df = pd.read_csv(params.CSV_DATA_SAMPLES, sep=';')

for index, row in df.iterrows():
    uid = UID = row['Uid']
    if "202" in UID:
        uid = "form-app-users"
    folder = row['Folder Name']
    if UID == "MJQ296DCcN" and folder == "2020-11-26-17_00_54_657915":
        continue

    breath = row['Breath filename']
    split = row['split']
    label = row['label']

    if split == 0 and label == 1:
        fold = "train_asthma_id"
    elif split == 0 and label == 0:
        fold = "train_health_id"
    elif split == 1 and label == 1:
        fold = "vad_asthma_id"
    elif split == 1 and label == 0:
        fold = "vad_health_id"
    elif split == 2 and label == 1:
        fold = "test_asthma_id"
    else:
        fold = "test_health_id"

    if UID not in user_all[fold]:
        user_all[fold].append(UID)


for f in user_all:
    print(f, len(user_all[f]))
f = open("audio_0124_asthma_users.pk", "wb")
joblib.dump(user_all, f)
f.close()
