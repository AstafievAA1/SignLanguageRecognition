import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from utils import load_config


class GestureDataset(Dataset):

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def load_raw_data(config: dict) -> Tuple[List, List]:
    data_path = config['data']['raw_path']
    actions = config['gestures']['actions']
    sequence_length = config['collection']['sequence_length']

    label_map = {action: idx for idx, action in enumerate(actions)}
    sequences = []
    labels = []

    print("\n Загрузка данных...")

    for action in actions:
        action_path = os.path.join(data_path, action)

        if not os.path.exists(action_path):
            print(f"   Папка {action} не найдена, пропускаем")
            continue

        sequence_folders = sorted([
            f for f in os.listdir(action_path)
            if os.path.isdir(os.path.join(action_path, f))
        ], key=lambda x: int(x))

        loaded_sequences = 0

        for seq_folder in sequence_folders:
            seq_path = os.path.join(action_path, seq_folder)

            window = []
            for frame_num in range(sequence_length):
                frame_path = os.path.join(seq_path, f"{frame_num}.npy")

                if not os.path.exists(frame_path):
                    break

                keypoints = np.load(frame_path)
                window.append(keypoints)

            if len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])
                loaded_sequences += 1

        print(f"  {action}: загружено {loaded_sequences} последовательностей")

    print(f"\n  Всего загружено: {len(sequences)} последовательностей")
    return sequences, labels

def augment_sequence(sequence: np.ndarray, label: int, actions: list) -> List[Tuple[np.ndarray, int]]:
    augmented = []

    # 1. Оригинал
    augmented.append((sequence.copy(), label))

    # 2. Горизонтальное зеркало со свапом lh <-> rh
    flipped = sequence.copy()
    for i in range(len(flipped)):
        frame = flipped[i]
        lh = frame[:63].reshape(21, 3).copy()
        rh = frame[63:].reshape(21, 3).copy()

        lh[:, 0] = -lh[:, 0]
        rh[:, 0] = -rh[:, 0]

        flipped[i] = np.concatenate([rh.flatten(), lh.flatten()])

    mirror_map = {}
    if 'turn_left' in actions and 'turn_right' in actions:
        mirror_map[actions.index('turn_left')] = actions.index('turn_right')
        mirror_map[actions.index('turn_right')] = actions.index('turn_left')
    flipped_label = mirror_map.get(label, label)
    augmented.append((flipped, flipped_label))

    # 3. Шум
    noisy = sequence + np.random.normal(0, 0.02, sequence.shape)
    augmented.append((noisy, label))

    return augmented


def augment_dataset(
    sequences: np.ndarray,
    labels: np.ndarray,
    augment_factor: int = 3,
    actions: list = None
) -> Tuple[np.ndarray, np.ndarray]:
    augmented_sequences = []
    augmented_labels = []

    for seq, label in zip(sequences, labels):
        aug_pairs = augment_sequence(seq, label, actions or [])

        for aug_seq, aug_label in aug_pairs[:augment_factor]:
            augmented_sequences.append(aug_seq)
            augmented_labels.append(aug_label)

    return np.array(augmented_sequences), np.array(augmented_labels)


def prepare_dataset(
    config_path: str = "../configs/config.yaml",
    augment: bool = True,
) -> dict:
    config = load_config(config_path)

    print("\n" + "=" * 70)
    print("ПОДГОТОВКА ДАТАСЕТА")
    print("=" * 70)

    sequences, labels = load_raw_data(config)

    if len(sequences) == 0:
        print("Нет данных для обработки!")
        return None

    sequences = np.array(sequences)
    labels = np.array(labels)

    print(f"\n  Форма данных: {sequences.shape}")
    print(f"  Последовательностей: {sequences.shape[0]}")
    print(f"  Длина последовательности: {sequences.shape[1]}")
    print(f"  Признаков: {sequences.shape[2]}")

    val_split = config['training']['validation_split']
    test_split = config['training']['test_split']

    # =========================================================
    #

    print("\n Разделение оригинальных данных (до аугментации)...")

    X_temp, X_test, y_temp, y_test = train_test_split(
        sequences, labels,
        test_size=test_split,
        random_state=42,
        stratify=labels
    )

    val_size = val_split / (1 - test_split)
    X_train_raw, X_val, y_train_raw, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=42,
        stratify=y_temp
    )

    print(f"  Оригиналы — Train: {len(X_train_raw)} | Val: {len(X_val)} | Test: {len(X_test)}")

    if augment:
        X_train, y_train = augment_dataset(
            X_train_raw, y_train_raw,
            augment_factor=3,
            actions=config['gestures']['actions']
        )
        print(f"  Train до аугментации:  {len(X_train_raw)}")
        print(f"  Train после аугментации: {len(X_train)}")
        print(f"  Val и Test — без аугментации (чистые оригиналы)")
    else:
        X_train, y_train = X_train_raw, y_train_raw

    print(f"\n  Итоговое разделение:")
    print(f"  Train: {len(X_train)} примеров")
    print(f"  Val:   {len(X_val)} примеров")
    print(f"  Test:  {len(X_test)} примеров")

    output_path = os.path.join(
        config['data']['processed_path'],
        config['data']['dataset_name']
    )

    np.savez(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        actions=config['gestures']['actions']
    )

    print(f"\n Данные сохранены: {output_path}")

    print("\n  Распределение классов (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for action_idx, count in zip(unique, counts):
        action_name = config['gestures']['actions'][action_idx]
        display_name = config['gestures']['display_names'][action_name]
        print(f"    {display_name}: {count} примеров")

    print("=" * 70 + "\n")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'actions': config['gestures']['actions'],
        'config': config
    }


def create_dataloaders(data_dict: dict, batch_size: int = 32) -> Tuple:
    train_dataset = GestureDataset(data_dict['X_train'], data_dict['y_train'])
    val_dataset = GestureDataset(data_dict['X_val'], data_dict['y_val'])
    test_dataset = GestureDataset(data_dict['X_test'], data_dict['y_test'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    data_dict = prepare_dataset(
        config_path="../configs/config.yaml",
        augment=True,
    )

    if data_dict:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dict,
            batch_size=32
        )

        print(" DataLoaders созданы:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches:   {len(val_loader)}")
        print(f"  Test batches:  {len(test_loader)}")

        X_batch, y_batch = next(iter(train_loader))
        print(f"\n  Пример батча:")
        print(f"  X shape: {X_batch.shape}")
        print(f"  y shape: {y_batch.shape}")