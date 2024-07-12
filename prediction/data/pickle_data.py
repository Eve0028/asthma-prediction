# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 11:25:58 2021

@author: xiatong
"""

import joblib
import librosa
import numpy as np
import pandas as pd

import data_params as params

SR = 16000  # sample rate
# SR_VGG = 16000  # VGG pretrained model sample rate
FRAME_LEN = int(SR / 10)  # 100 ms
HOP = int(FRAME_LEN / 2)  # 50%overlap, 5ms


def get_feature(file):
    y, sr = librosa.load(file, sr=SR, mono=True, offset=0.0, duration=None)
    yt, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
    yt_n = yt / np.max(np.abs(yt))  # normolized the sound
    return yt_n


data_all_asthma = {}
data_all_health = {}

path = params.DATA_SAMPLES

df = pd.read_csv(params.CSV_DATA_SAMPLES, sep=';')

for index, row in df.iterrows():
    uid = UID = row['Uid']
    if "202" in uid:
        uid = "form-app-users"
    folder = row['Folder Name']
    if UID == "MJQ296DCcN" and folder == "2020-11-26-17_00_54_657915":
        continue

    print(UID, "===", folder)
    breath = row['Breath filename']
    split = row['split']
    label = row['label']
    fname_b = "/".join([path, uid, folder, breath])

    if label == 1:
        if UID not in data_all_asthma:
            data_all_asthma[UID] = [
                {
                    "breath": get_feature(fname_b),
                    "label": label,
                }
            ]
        else:
            data_all_asthma[UID].append(
                {
                    "breath": get_feature(fname_b),
                    "label": label,
                }
            )
    if label == 0:
        if UID not in data_all_health:
            data_all_health[UID] = [
                {
                    "breath": get_feature(fname_b),
                    "label": label,
                }
            ]
        else:
            data_all_health[UID].append(
                {
                    "breath": get_feature(fname_b),
                    "label": label,
                }
            )


f = open("audio_0124_asthma.pk", "wb")
joblib.dump(data_all_asthma, f)
f.close()

f = open("audio_0124_health.pk", "wb")
joblib.dump(data_all_health, f)
f.close()
