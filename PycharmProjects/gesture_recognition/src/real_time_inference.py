import cv2
import numpy as np
import torch
import mediapipe as mp
import time
from collections import deque
from typing import Optional
import os
import sys

from model import GestureLSTM
from utils import (
    load_config,
    get_device,
    extract_keypoints,
    mediapipe_detection,
    draw_styled_landmarks,
    prob_viz,
    display_info_panel
)
from stateful_lstm_wrapper import StatefulLSTMInference

class StatefulGestureRecognizer:
    def __init__(
            self,
            trained_model,
            config,
            device='cpu',
    ):
        self.stateful_model = StatefulLSTMInference(trained_model, device)

        self.config = config
        self.actions = config['gestures']['actions']
        self.display_names = config['gestures']['display_names']
        self.device = device

        self.min_frames_warmup = config['inference']['min_detection_frames']
        self.confidence_threshold = config['inference']['confidence_threshold']
        self.smoothing_window = config['inference']['smoothing_window']
        self.reset_timeout = 1.0

        self.frame_count = 0
        self.last_detection_time = time.time()
        self.current_gesture = None
        self.current_probs = None

        self.predictions_buffer = deque(maxlen=self.smoothing_window)

        self.gesture_history = deque(maxlen=config['inference']['max_history'])


        self._gesture_display_timer = 0.0
        self._gesture_display_duration = 2.0

        print(" Stateful Gesture Recognizer инициализирован")
        print(f"  Min warmup frames: {self.min_frames_warmup}")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Smoothing window: {self.smoothing_window}")
        print(f"  Reset timeout: {self.reset_timeout}s")

    def process_frame(self, results) -> tuple:

        now = time.time()

        if not results.multi_hand_landmarks:
            if now - self.last_detection_time > self.reset_timeout:
                self._reset_lstm()

            if now - self._gesture_display_timer > self._gesture_display_duration:
                self.current_gesture = None
            return None, self.current_probs

        self.last_detection_time = now

        keypoints = extract_keypoints(results)
        predicted_class, probabilities = self.stateful_model.predict_single_frame(keypoints)
        self.current_probs = probabilities

        self.frame_count += 1
        self.predictions_buffer.append(predicted_class)

        if self.frame_count < self.min_frames_warmup:
            return None, probabilities

        recent = list(self.predictions_buffer)
        unique, counts = np.unique(recent, return_counts=True)
        smoothed_class = unique[np.argmax(counts)]
        confidence = probabilities[smoothed_class]

        if confidence > self.confidence_threshold:
            gesture = self.actions[smoothed_class]
            display_name = self.display_names[gesture]

            if gesture != 'no_gesture':
                if not self.gesture_history or self.gesture_history[-1] != gesture:
                    self.gesture_history.append(gesture)
                    print(f" Detected: {display_name} ({confidence:.2%})")

            self.current_gesture = gesture
            self._gesture_display_timer = now
            self._reset_lstm()

            return gesture, probabilities

        return None, probabilities

    def _reset_lstm(self):

        self.stateful_model.reset_state()
        self.frame_count = 0
        self.predictions_buffer.clear()

    def reset(self):
        self._reset_lstm()
        self.current_gesture = None
        self.current_probs = None
        print(" Полный сброс состояния")


def find_latest_model(models_path: str) -> Optional[str]:
    if not os.path.exists(models_path):
        return None
    models = [f for f in os.listdir(models_path) if f.endswith('.pth')]
    if not models:
        return None
    models_with_time = [
        (f, os.path.getmtime(os.path.join(models_path, f)))
        for f in models
    ]
    models_with_time.sort(key=lambda x: x[1], reverse=True)
    return os.path.join(models_path, models_with_time[0][0])


def run_stateful_recognition(model_path: str, config_path: str = "../configs/config.yaml"):
    config = load_config(config_path)
    device = get_device()

    print(f"\n Загрузка модели: {model_path}")

    if not os.path.exists(model_path):
        print(f" Модель не найдена: {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    model = GestureLSTM(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_classes=len(config['gestures']['actions']),
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional'],
        use_attention=config['model'].get('use_attention', True)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f" Модель загружена:")
    print(f"  Эпоха: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Accuracy: {checkpoint.get('val_acc', 0.0):.4f}")

    recognizer = StatefulGestureRecognizer(
        trained_model=model,
        config=config,
        device=device,
    )

    cap = cv2.VideoCapture(config['camera']['camera_id'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(" Ошибка: не удалось открыть камеру")
        return

    print(f"\n Камера открыта: {config['camera']['width']}×{config['camera']['height']}")

    mp_hands = mp.solutions.hands
    fps_counter = deque(maxlen=30)
    frame_num = 0

    print("\n" + "=" * 70)
    print("Управление: 'q' — выход | 'r' — сброс состояния | 'c' — очистить историю")
    print("=" * 70 + "\n")

    with mp_hands.Hands(
            min_detection_confidence=config['mediapipe']['min_detection_confidence'],
            min_tracking_confidence=config['mediapipe']['min_tracking_confidence'],
            max_num_hands=config['mediapipe']['max_num_hands'],
            model_complexity=config['mediapipe']['model_complexity']
    ) as hands:

        while cap.isOpened():
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print(" Ошибка чтения кадра")
                break

            image, results = mediapipe_detection(frame, hands)
            gesture, probs = recognizer.process_frame(results)

            if config['visualization']['show_landmarks']:
                draw_styled_landmarks(image, results)

            if probs is not None and config['visualization']['show_probabilities']:
                image = prob_viz(
                    probs,
                    recognizer.actions,
                    recognizer.display_names,
                    image,
                    config['visualization']['colors']
                )

            fps = 1.0 / (time.time() - start_time + 1e-9)
            fps_counter.append(fps)
            avg_fps = np.mean(fps_counter)

            image = display_info_panel(
                image,
                recognizer.current_gesture,
                recognizer.display_names,
                avg_fps,
                list(recognizer.gesture_history) if config['visualization']['show_history'] else []
            )

            cv2.putText(
                image,
                f"Frames: {recognizer.frame_count} | Warmup: {recognizer.min_frames_warmup}",
                (10, image.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            cv2.putText(
                image, "Stateful LSTM Mode",
                (10, image.shape[0] - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )

            cv2.imshow('Stateful LSTM Gesture Recognition', image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n Выход...")
                break
            elif key == ord('r'):
                recognizer.reset()
            elif key == ord('c'):
                recognizer.gesture_history.clear()
                print(" История очищена")

            frame_num += 1

            if frame_num % 100 == 0:
                print(
                    f"  Frames: {frame_num} | FPS: {avg_fps:.1f} | "
                    f"Gestures: {len(recognizer.gesture_history)}"
                )

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print("РАСПОЗНАВАНИЕ ЗАВЕРШЕНО")
    print("=" * 70)
    print(f"Обработано фреймов: {frame_num}")
    print(f"Средний FPS: {np.mean(fps_counter):.1f}")
    print(f"Распознано жестов: {len(recognizer.gesture_history)}")

    if recognizer.gesture_history:
        print("\nИстория жестов:")
        for i, g in enumerate(recognizer.gesture_history, 1):
            print(f"  {i}. {recognizer.display_names[g]}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    config = load_config("../configs/config.yaml")

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = find_latest_model(config['data']['models_path'])
        if model_path is None:
            print(" Обученная модель не найдена!")
            print("Использование: python real_time_inference_stateful.py <path_to_model.pth>")
            sys.exit(1)
        print(f" Найдена модель: {model_path}")

    run_stateful_recognition(model_path)