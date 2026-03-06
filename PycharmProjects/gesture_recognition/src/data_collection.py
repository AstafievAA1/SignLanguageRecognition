import cv2
import numpy as np
import os
import mediapipe as mp
from utils import (
    load_config,
    extract_keypoints,
    mediapipe_detection,
    draw_styled_landmarks,
    create_directories
)

def check_existing_data(config: dict) -> dict:
    data_path = config['data']['raw_path']
    actions = config['gestures']['actions']
    sequence_length = config['collection']['sequence_length']

    existing_data = {}

    for action in actions:
        action_path = os.path.join(data_path, action)

        if not os.path.exists(action_path):
            existing_data[action] = {
                'count': 0, 'max_folder': -1,
                'folders': [], 'complete': [], 'incomplete': []
            }
            continue

        folders = [
            f for f in os.listdir(action_path)
            if os.path.isdir(os.path.join(action_path, f)) and f.isdigit()
        ]

        if not folders:
            existing_data[action] = {
                'count': 0, 'max_folder': -1,
                'folders': [], 'complete': [], 'incomplete': []
            }
            continue

        folders_int = sorted([int(f) for f in folders])
        complete_folders = []
        incomplete_folders = []

        for folder_num in folders_int:
            folder_path = os.path.join(action_path, str(folder_num))
            frames = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

            if len(frames) == sequence_length:
                complete_folders.append(folder_num)
            else:
                incomplete_folders.append((folder_num, len(frames)))

        existing_data[action] = {
            'count': len(complete_folders),
            'max_folder': max(folders_int),
            'folders': folders_int,
            'complete': complete_folders,
            'incomplete': incomplete_folders
        }

    return existing_data


def display_existing_data(config: dict, existing_data: dict) -> bool:
    actions = config['gestures']['actions']
    display_names = config['gestures']['display_names']

    print("\n" + "=" * 70)
    print(" СУЩЕСТВУЮЩИЕ ДАННЫЕ В ПАПКАХ")
    print("=" * 70)

    total_videos = 0
    has_data = False

    for action in actions:
        data = existing_data[action]
        display_name = display_names[action]

        if data['count'] > 0 or data['incomplete']:
            has_data = True
            total_videos += data['count']

            print(f"\n📁 {display_name} ({action}):")
            if data['count'] > 0:
                print(f"    Полных видео: {data['count']}/{config['collection']['num_sequences']}")
                print(f"    Номера папок: {data['complete'][:10]}", end="")
                if len(data['complete']) > 10:
                    print(f" ... и еще {len(data['complete']) - 10}")
                else:
                    print()

            if data['incomplete']:
                print(f"   ️  Неполных видео: {len(data['incomplete'])}")
                for folder_num, frames in data['incomplete'][:5]:
                    print(f"      Папка {folder_num}: {frames}/{config['collection']['sequence_length']} фреймов")
                if len(data['incomplete']) > 5:
                    print(f"      ... и еще {len(data['incomplete']) - 5}")
        else:
            print(f"\n {display_name} ({action}): нет данных")

    print("\n" + "=" * 70)
    if has_data:
        print(f" Всего записано полных видео: {total_videos}")
        print(f" Цель: {len(actions) * config['collection']['num_sequences']} видео")
        progress = (total_videos / (len(actions) * config['collection']['num_sequences'])) * 100
        print(f" Прогресс: {progress:.1f}%")
    else:
        print(" Нет записанных данных. Начнем с нуля!")
    print("=" * 70)

    return has_data


def get_start_folder(config: dict, existing_data: dict) -> int:
    max_folder = -1
    for action, data in existing_data.items():
        if data['max_folder'] > max_folder:
            max_folder = data['max_folder']
    start_folder = max_folder + 1
    print(f"\n Запись будет продолжена с папки №{start_folder}")
    return start_folder


def collect_gesture_data(config_path: str = "../configs/config.yaml"):
    config = load_config(config_path)
    create_directories(config)

    existing_data = check_existing_data(config)
    has_existing = display_existing_data(config, existing_data)

    actions = config['gestures']['actions']
    display_names = config['gestures']['display_names']
    num_sequences = config['collection']['num_sequences']
    sequence_length = config['collection']['sequence_length']
    countdown_frames = config['collection']['countdown_frames']
    data_path = config['data']['raw_path']

    if has_existing:
        start_folder = get_start_folder(config, existing_data)
        print(f"\n Найдены существующие данные. Запись продолжится с видео №{start_folder}")
        response = input("\nПродолжить с этого номера? (y/n) или введите свой номер: ").strip()
        if response.lower() == 'n':
            print(" Отменено")
            return
        elif response.isdigit():
            start_folder = int(response)
            print(f" Начнем с папки №{start_folder}")
    else:
        start_folder = config['collection']['start_folder']

    print("\n" + "=" * 70)
    print(" ВЫБОР ЖЕСТА ДЛЯ ЗАПИСИ")
    print("=" * 70)

    for i, action in enumerate(actions, 1):
        display_name = display_names[action]
        existing_count = existing_data[action]['count']
        print(f"{i:2}. {display_name} ({action}) - уже есть {existing_count} видео")

    print("=" * 70)

    while True:
        gesture_choice = input("\nВыберите жест для записи (введите номер или 'all' для всех): ").strip()
        if gesture_choice.lower() == 'all':
            selected_actions = actions
            print(" Будет записано ВСЕ жесты")
            break
        elif gesture_choice.isdigit():
            idx = int(gesture_choice) - 1
            if 0 <= idx < len(actions):
                selected_actions = [actions[idx]]
                print(f" Выбран жест: {display_names[actions[idx]]} ({actions[idx]})")
                break
            else:
                print(f" Пожалуйста, введите число от 1 до {len(actions)}")
        else:
            print(" Пожалуйста, введите номер жеста или 'all'")

    print("\n" + "=" * 70)
    print(f" Запись с папки №{start_folder} | {num_sequences} видео | {sequence_length} фреймов")
    print("=" * 70)
    print("\nИнструкции:")
    print("- Нажмите 's' чтобы пропустить текущее видео")
    print("- Нажмите 'q' чтобы выйти")
    print("- Держите обе руки в кадре во время записи")
    print("=" * 70 + "\n")

    input("Нажмите Enter чтобы начать...")

    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(config['camera']['camera_id'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])

    if not cap.isOpened():
        print(" Ошибка: не удалось открыть камеру")
        return

    print(" Камера открыта")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = config['camera']['fps']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    with mp_hands.Hands(
            min_detection_confidence=config['mediapipe']['min_detection_confidence'],
            min_tracking_confidence=config['mediapipe']['min_tracking_confidence'],
            max_num_hands=config['mediapipe']['max_num_hands'],
            model_complexity=config['mediapipe']['model_complexity']
    ) as hands:

        for action_idx, action in enumerate(selected_actions):
            print(f"\n{'=' * 70}")
            print(f"ЖЕСТ {action_idx + 1}/{len(selected_actions)}: {display_names[action]}")
            print(f"Уже записано: {existing_data[action]['count']} видео")
            print(f"{'=' * 70}")

            for sequence in range(start_folder, start_folder + num_sequences):
                print(f"\n Видео {sequence - start_folder + 1}/{num_sequences} (папка №{sequence})")

                sequence_path = os.path.join(data_path, action, str(sequence))
                os.makedirs(sequence_path, exist_ok=True)

                video_path = os.path.join(sequence_path, f"video_{sequence}.mp4")
                video_writer = None
                frame_count = 0
                recording = False
                countdown = countdown_frames
                skip_video = False

                while frame_count < sequence_length:
                    ret, frame = cap.read()
                    if not ret:
                        print(" Ошибка чтения кадра")
                        break

                    image, results = mediapipe_detection(frame, hands)
                    draw_styled_landmarks(image, results)

                    if not recording:
                        if countdown > 0:
                            seconds_left = (countdown // fps) + 1
                            cv2.putText(
                                image, f"Get ready: {seconds_left}",
                                (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                2, (0, 255, 0), 3, cv2.LINE_AA
                            )
                            countdown -= 1
                        else:
                            recording = True
                            video_writer = cv2.VideoWriter(
                                video_path, fourcc, fps, (frame_width, frame_height)
                            )
                            print(" Запись началась!")

                    if recording:
                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(sequence_path, f"{frame_count}.npy")
                        np.save(npy_path, keypoints)

                        if video_writer is not None:
                            video_writer.write(frame)

                        cv2.putText(
                            image, f"Record: {frame_count + 1}/{sequence_length}",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 3, cv2.LINE_AA
                        )
                        frame_count += 1

                    cv2.rectangle(image, (0, 0), (image.shape[1], 60), (245, 117, 16), -1)
                    cv2.putText(
                        image, f"Gesture: {display_names[action]}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA
                    )

                    progress_text = (
                        f"Gesture {action_idx + 1}/{len(selected_actions)} | "
                        f"Video {sequence - start_folder + 1}/{num_sequences}"
                    )
                    cv2.putText(
                        image, progress_text,
                        (image.shape[1] - 500, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA
                    )

                    cv2.imshow('Gesture data collection', image)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n️ Выход...")
                        if video_writer is not None:
                            video_writer.release()
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    elif key == ord('s'):
                        print(" Видео пропущено")
                        skip_video = True
                        break

                if video_writer is not None:
                    video_writer.release()

                if skip_video:
                    continue

                print(f" Видео {sequence - start_folder + 1} сохранено в папку {sequence}")

                for i in range(2):
                    ret, frame = cap.read()
                    if ret:
                        cv2.putText(
                            frame, "Get ready for the next video...",
                            (100, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 0), 3, cv2.LINE_AA
                        )
                        cv2.imshow('Gesture data collection', frame)
                        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print(" СБОР ДАННЫХ ЗАВЕРШЕН!")
    print("=" * 70)
    print(f"Данные сохранены в: {data_path}")
    print(f"Собрано жестов: {len(selected_actions)}")
    print(f"Видео на жест: {num_sequences}")

    existing_data = check_existing_data(config)
    display_existing_data(config, existing_data)


def verify_collected_data(config_path: str = "../configs/config.yaml"):
    config = load_config(config_path)
    data_path = config['data']['raw_path']
    actions = config['gestures']['actions']
    display_names = config['gestures']['display_names']
    sequence_length = config['collection']['sequence_length']

    print("\n" + "=" * 70)
    print(" ПРОВЕРКА СОБРАННЫХ ДАННЫХ")
    print("=" * 70)

    total_sequences = 0
    total_complete = 0
    total_incomplete = 0
    total_videos = 0

    for action in actions:
        action_path = os.path.join(data_path, action)
        display_name = display_names[action]

        if not os.path.exists(action_path):
            print(f" {display_name}: папка не найдена")
            continue

        sequences = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]
        if not sequences:
            print(f" {display_name}: нет данных")
            continue

        num_seq = len(sequences)
        total_sequences += num_seq

        complete = 0
        incomplete = 0
        videos_count = 0

        for seq in sequences:
            seq_path = os.path.join(action_path, seq)
            frames = len([f for f in os.listdir(seq_path) if f.endswith('.npy')])
            videos = len([f for f in os.listdir(seq_path) if f.endswith('.mp4')])

            if frames == sequence_length:
                complete += 1
            else:
                incomplete += 1

            videos_count += videos

        total_complete += complete
        total_incomplete += incomplete
        total_videos += videos_count

        status = "+" if incomplete == 0 else "-"
        print(f"{status} {display_name}: {complete} полных + {incomplete} неполных = {num_seq} всего")

    print("=" * 70)
    print(f"   Полных: {total_complete} | Неполных: {total_incomplete} | Видео файлов: {total_videos}")

    if total_incomplete > 0:
        print(f"\n  ВНИМАНИЕ: Есть {total_incomplete} неполных видео!")
    else:
        print(f"\n Все данные в порядке!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "collect"

    if mode == "collect":
        collect_gesture_data()
    elif mode == "verify":
        verify_collected_data()
    else:
        print("Использование:")
        print("  python data_collection.py collect  - сбор данных")
        print("  python data_collection.py verify   - проверка данных")