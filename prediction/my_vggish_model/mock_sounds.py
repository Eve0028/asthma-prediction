import numpy as np
import joblib
import random


# Function to simulate audio feature extraction with varying lengths
def mock_get_feature(duration, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)  # A 440 Hz tone
    yt_n = y / np.max(np.abs(y))  # normalized the sound
    return yt_n


# UID generator
def uid_generator(start=1):
    uid = start
    while True:
        yield f"{uid:03}"
        uid += 1


# Duration generator
def duration_generator(min_duration=1.0, max_duration=5.0):
    while True:
        yield random.uniform(min_duration, max_duration)


# Initialize dictionaries to store mock data
data_all_asthma = {}
data_all_health = {}


# Helper function to populate data
def populate_data(uid_gen, duration_gen, num_samples, label, data_dict):
    for _ in range(num_samples):
        uid = next(uid_gen)
        duration = next(duration_gen)
        # data_dict[uid] = [{"breath": mock_get_feature(duration), "label": label}]
        data_dict[uid] = mock_get_feature(duration)


# Number of samples
num_asthma_samples = 10
num_health_samples = 10

# Create UID and duration generators
asthma_uid_gen = uid_generator(start=1)
health_uid_gen = uid_generator(start=101)
duration_gen = duration_generator()

# Populate mock data
populate_data(asthma_uid_gen, duration_gen, num_asthma_samples, 1, data_all_asthma)
populate_data(health_uid_gen, duration_gen, num_health_samples, 0, data_all_health)

# Save the mocked data to files
with open("audio_0124_asthma.pk", "wb") as f:
    joblib.dump(data_all_asthma, f)

with open("audio_0124_health.pk", "wb") as f:
    joblib.dump(data_all_health, f)
