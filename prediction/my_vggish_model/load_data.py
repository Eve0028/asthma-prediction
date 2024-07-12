import joblib
import numpy as np
from sklearn.model_selection import train_test_split

import params


def sets_samples_counts(samples_count, ratios):
    """Calculate the number of samples for each set based on the ratios."""
    train_ratio, val_ratio, test_ratio = ratios
    train_class_cnt = int(samples_count * train_ratio)
    val_class_cnt = int(samples_count * val_ratio)
    test_class_cnt = int(samples_count * test_ratio)
    return train_class_cnt, val_class_cnt, test_class_cnt


def load_data(data_path, samples_count=None, ratios=(0.7, 0.2, 0.1)):
    """Load data for training, validation, and testing."""
    print("Start to load data:", data_path)

    data_asthma = joblib.load(open(data_path + "_asthma.pk", "rb"))  # Load positive samples
    data_health = joblib.load(open(data_path + "_health.pk", "rb"))  # Load negative samples

    # Extract UIDs for asthma and health
    asthma_uids = list(data_asthma.keys())
    health_uids = list(data_health.keys())

    # Determine sample counts for each set
    total_samples = min(len(asthma_uids), len(health_uids))
    if samples_count:
        train_class_cnt, val_class_cnt, test_class_cnt = sets_samples_counts(samples_count // 2, ratios)
    elif hasattr(params, 'SAMPLES_COUNT'):
        train_class_cnt, val_class_cnt, test_class_cnt = sets_samples_counts(params.SAMPLES_COUNT // 2, ratios)
    else:
        train_class_cnt, val_class_cnt, test_class_cnt = sets_samples_counts(total_samples, ratios)

    def create_task_set(uids, class_cnt, data, label):
        """Helper function to create a task set."""
        task = []
        for uid in uids[:class_cnt]:
            # task.append({"breath": data[uid][0]["breath"], "label": label})
            task.append({"breath": data[uid], "label": label})
        return task

    # Split UIDs into training, validation, and test sets
    train_asthma, temp_asthma = train_test_split(asthma_uids, train_size=ratios[0])
    val_asthma, test_asthma = train_test_split(temp_asthma, test_size=ratios[2] / (ratios[1] + ratios[2]))

    train_health, temp_health = train_test_split(health_uids, train_size=ratios[0])
    val_health, test_health = train_test_split(temp_health, test_size=ratios[2] / (ratios[1] + ratios[2]))

    # Create training, validation, and testing sets
    train_task = create_task_set(train_asthma, train_class_cnt, data_asthma, [0, 1])
    train_task += create_task_set(train_health, train_class_cnt, data_health, [1, 0])
    print("Train: asthma:", len(train_asthma), "health:", len(train_health))

    vad_task = create_task_set(val_asthma, val_class_cnt, data_asthma, [0, 1])
    vad_task += create_task_set(val_health, val_class_cnt, data_health, [1, 0])
    print("Validation: asthma:", len(val_asthma), "health:", len(val_health))

    test_task = create_task_set(test_asthma, test_class_cnt, data_asthma, [0, 1])
    test_task += create_task_set(test_health, test_class_cnt, data_health, [1, 0])
    print("Test: asthma:", len(test_asthma), "health:", len(test_health))

    # Convert lists to numpy arrays for consistency
    train_task = np.array(train_task, dtype=object)
    vad_task = np.array(vad_task, dtype=object)
    test_task = np.array(test_task, dtype=object)

    # Shuffle samples
    rng = np.random.default_rng(222)
    rng.shuffle(train_task)
    rng.shuffle(vad_task)
    rng.shuffle(test_task)

    return train_task, vad_task, test_task
