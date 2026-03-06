import cv2
import numpy as np
import yaml
import os
import torch
import mediapipe as mp
from typing import Dict, List, Tuple, Optional


def load_config(config_path: str = "../configs/config.yaml") -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Используется Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Используется NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Используется CPU")
    return device


def create_directories(config: Dict) -> None:
    dirs = [
        config['data']['raw_path'],
        config['data']['processed_path'],
        config['data']['models_path']
    ]
    for action in config['gestures']['actions']:
        action_dir = os.path.join(config['data']['raw_path'], action)
        dirs.append(action_dir)
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print(f" Созданы директории для {len(config['gestures']['actions'])} жестов")


def extract_keypoints(results) -> np.ndarray:
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = handedness.classification[0].label
            keypoints = np.array([
                [res.x, res.y, res.z] for res in hand_landmarks.landmark
            ]).flatten()

            if label == 'Left':
                lh = keypoints
            else:
                rh = keypoints

    return np.concatenate([lh, rh])


def mediapipe_detection(image: np.ndarray, model) -> Tuple[np.ndarray, any]:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image: np.ndarray, results) -> np.ndarray:
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
    return image


def prob_viz(
    predictions: np.ndarray,
    actions: List[str],
    display_names: Dict[str, str],
    image: np.ndarray,
    colors: Dict[str, List[int]]
) -> np.ndarray:
    output_frame = image.copy()

    for num, prob in enumerate(predictions):
        action = actions[num]
        color = tuple(colors.get(action, [255, 255, 255]))

        cv2.rectangle(
            output_frame,
            (0, 60 + num * 40),
            (int(prob * 300), 90 + num * 40),
            color,
            -1
        )

        display_name = display_names.get(action, action)
        cv2.putText(
            output_frame,
            f"{display_name}: {prob:.2f}",
            (5, 85 + num * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return output_frame


def smooth_predictions(predictions_history: List[int], window_size: int = 10) -> Optional[int]:
    if len(predictions_history) < window_size:
        return None
    recent = predictions_history[-window_size:]
    unique, counts = np.unique(recent, return_counts=True)
    max_count = np.max(counts)
    if max_count > window_size // 2:
        return unique[np.argmax(counts)]
    return None


def display_info_panel(
    image: np.ndarray,
    current_gesture: str,
    display_names: Dict[str, str],
    fps: float,
    history: List[str]
) -> np.ndarray:
    h, w = image.shape[:2]

    cv2.rectangle(image, (0, 0), (w, 50), (245, 117, 16), -1)

    if current_gesture:
        display_name = display_names.get(current_gesture, current_gesture)
        text = f"Gesture: {display_name}"
    else:
        text = "Waiting for a gesture..."

    cv2.putText(image, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"FPS: {fps:.1f}", (w - 150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if history:
        cv2.rectangle(image, (0, h - 50), (w, h), (50, 50, 50), -1)
        history_text = " --> ".join([display_names.get(g, g) for g in history[-5:]])
        cv2.putText(
            image, f"History: {history_text}",
            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )

    return image


def print_training_info(
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_acc: float,
    val_acc: float,
    lr: float
) -> None:
    print(
        f"Epoch {epoch:3d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.2%} | "
        f"Val Acc: {val_acc:.2%} | "
        f"LR: {lr:.6f}"
    )


if __name__ == "__main__":
    config = load_config()
    print("Конфигурация загружена:")
    print(f"Жестов: {len(config['gestures']['actions'])}")
    print(f"Жесты: {config['gestures']['actions']}")
    device = get_device()
    print(f"Устройство: {device}")