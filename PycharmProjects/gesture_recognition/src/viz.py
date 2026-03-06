
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib

matplotlib.use('Qt5Agg')  # PyCharm friendly


def denormalize_keypoints(kp_norm: np.ndarray, img_w: int = 640, img_h: int = 480) -> np.ndarray:
    """[0,1] → ПИКСЕЛИ! Z×200 для видимости"""
    kp = kp_norm.copy().reshape(42, 3)
    kp[:, 0] *= img_w  # X: 0.48 → 307px
    kp[:, 1] *= img_h  # Y: 0.52 → 250px
    kp[:, 2] *= 200  # 🔥 Z: -0.01 → -2.0см
    return kp.flatten()


def load_raw_sequence_px(seq_path: str, img_w: int = 640, img_h: int = 480):
    """ЧИТАЕТ .npy + ДЕНОРМАЛИЗУЕТ → ПИКСЕЛИ + Z×200"""
    frames_px = []
    for frame_num in range(45):
        frame_path = os.path.join(seq_path, f"{frame_num}.npy")
        if os.path.exists(frame_path):
            kp_norm = np.load(frame_path)
            kp_px = denormalize_keypoints(kp_norm, img_w, img_h)
            left_px = kp_px[:63].reshape(21, 3)
            right_px = kp_px[63:].reshape(21, 3)
            frame_3d = np.vstack([left_px, right_px])
            frames_px.append(frame_3d)
    return np.array(frames_px)


def create_single_animation(gesture_name: str, seq_folder: str, data_path: str = "data/raw"):
    gesture_path = os.path.join(data_path, gesture_name, seq_folder)
    frames_3d = load_raw_sequence_px(gesture_path)

    if len(frames_3d) == 0:
        print(f"❌ {gesture_name}/{seq_folder}: нет данных")
        return None

    print(f"✅ Загружено {len(frames_3d)} кадров из {gesture_name}/{seq_folder}")

    # 🔥 АНАЛИЗ диапазонов!
    x_all, y_all, z_all = frames_3d[:, :, 0], frames_3d[:, :, 1], frames_3d[:, :, 2]
    print(f"📏 Диапазоны:")
    print(f"   X: {x_all.min():.0f} → {x_all.max():.0f} px")
    print(f"   Y: {y_all.min():.0f} → {y_all.max():.0f} px")
    print(f"   Z: {z_all.min():+.1f} → {z_all.max():+.1f} см (×200)")

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 🔥 ПРАВИЛЬНЫЕ пропорции 3D!
    ax.set_box_aspect([1, 0.75, 0.3])  # X:Y:Z

    # Инициализация объектов
    left_wrist_line, = ax.plot([], [], [], 'blue', linewidth=6, label='Левая запястье', alpha=0.8)
    right_wrist_line, = ax.plot([], [], [], 'red', linewidth=6, label='Правая запястье', alpha=0.8)

    left_scatter = ax.scatter([], [], [], c='blue', s=150, alpha=0.9, edgecolors='black', linewidth=2)
    right_scatter = ax.scatter([], [], [], c='red', s=150, alpha=0.9, edgecolors='black', linewidth=2)

    def animate(frame_idx):
        frame = frames_3d[frame_idx]

        # Текущие точки рук
        left_hand = frame[:21]
        right_hand = frame[21:]

        left_scatter._offsets3d = (left_hand[:, 0], left_hand[:, 1], left_hand[:, 2])
        right_scatter._offsets3d = (right_hand[:, 0], right_hand[:, 1], right_hand[:, 2])

        # Траектория запястий (20 кадров)
        trail_len = min(frame_idx + 1, 20)
        left_trail = frames_3d[:frame_idx + 1, 0, :]
        right_trail = frames_3d[:frame_idx + 1, 21, :]

        left_wrist_line.set_data_3d(left_trail[-trail_len:, 0],
                                    left_trail[-trail_len:, 1],
                                    left_trail[-trail_len:, 2])
        right_wrist_line.set_data_3d(right_trail[-trail_len:, 0],
                                     right_trail[-trail_len:, 1],
                                     right_trail[-trail_len:, 2])

        # Δ перемещения
        delta_x = frames_3d[frame_idx, 0, 0] - frames_3d[0, 0, 0]
        delta_y = frames_3d[frame_idx, 0, 1] - frames_3d[0, 0, 1]
        delta_z = frames_3d[frame_idx, 0, 2] - frames_3d[0, 0, 2]

        wrist_pos = frames_3d[frame_idx, 0, :2]

        ax.set_title(f'{gesture_name} | #{seq_folder}\n'
                     f'Кадр {frame_idx + 1}/45 | ΔX={delta_x:+.0f}px ΔY={delta_y:+.0f}px ΔZ={delta_z:+.1f}см\n'
                     f'Запястье: ({wrist_pos[0]:.0f},{wrist_pos[1]:.0f})px | t={frame_idx * 0.033:.1f}с',
                     fontsize=14, fontweight='bold', pad=20)

        # 🔥 АДАПТИВНЫЕ оси под реальные данные!
        x_range = [max(0, x_all.min() - 50), min(700, x_all.max() + 50)]
        y_range = [max(0, y_all.min() - 50), min(550, y_all.max() + 50)]
        z_range = [z_all.min() - 5, z_all.max() + 5]

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)

        ax.set_xlabel('X ← ЛЕВО          640px         ПРАВО →', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y ↑ ВЕРХ         480px         НИЗ ↓', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z ← ДАЛЕКО      {z_all.min():+.0f}..{z_all.max():+.0f}см     БЛИЗКО →', fontsize=12,
                      fontweight='bold')

        # 🔥 ФИКС Z-меток!
        ax.zaxis.labelpad = 15

        return left_scatter, right_scatter, left_wrist_line, right_wrist_line

    anim = FuncAnimation(fig, animate, frames=45, interval=150, blit=False, repeat=True)

    plt.suptitle(f'🎬 3D ПИКСЕЛЬНАЯ АНИМАЦИЯ РЖЯ с Z — {gesture_name.upper()}',
                 fontsize=18, fontweight='bold', color='darkgreen')
    plt.tight_layout()

    print("\n🔥 Z-координата РАБОТАЕТ! Инструкции:")
    print("• ЛКМ + перетаскивание = ВРАЩАЙ ОСИ!")
    print("• Колёсико = ЗУМ (смотри Z!)")
    print("• ПКМ + перетаскивание = ПАН")
    print("• Закрой окно = следующий жест")

    plt.show(block=True)
    return anim


def interactive_gesture_viewer(data_path: str = "data/raw"):
    """ИНТЕРАКТИВНЫЙ просмотр ПИКСЕЛЬНЫХ РЖЯ с Z"""
    if not os.path.exists(data_path):
        print(f"❌ Папка {data_path} не найдена!")
        return

    gestures = sorted([d for d in os.listdir(data_path)
                       if os.path.isdir(os.path.join(data_path, d))])

    print("🎬 3D ПИКСЕЛЬНАЯ АНИМАЦИЯ РЖЯ с Z-координатой!")
    print("=" * 70)
    print("🔥 640x480px + Z×200см! Адаптивные оси! Вращай ЛКМ!")
    print("Доступные жесты:", ", ".join(gestures))
    print("\n📏 ΔX/ΔY в пикселях | ΔZ в сантиметрах\n")

    while True:
        gesture = input("Выбери жест (или 'exit'): ").strip()
        if gesture.lower() == 'exit':
            break

        if gesture not in gestures:
            print(f"❌ Жест '{gesture}' не найден! ({', '.join(gestures)})")
            continue

        gesture_path = os.path.join(data_path, gesture)
        sequences = sorted([d for d in os.listdir(gesture_path)
                            if os.path.isdir(os.path.join(gesture_path, d))],
                           key=lambda x: int(x))

        print(f"\n📁 {gesture}: {len(sequences)} последовательностей")
        print("💾 Последние 5: ", sequences[-5:])
        print(f"Номер (0-{len(sequences) - 1}): ", end='')

        try:
            seq_id = int(input())
            if 0 <= seq_id < len(sequences):
                seq_folder = sequences[seq_id]
                print(f"▶️ Запускаем {gesture}/{seq_folder}... (ПИКСЕЛИ + Z!)")
                create_single_animation(gesture, seq_folder)
            else:
                print(f"❌ 0-{len(sequences) - 1}")
        except ValueError:
            print("❌ ЦИФРУ!")

    print("\n✅ ПИКСЕЛЬНАЯ 3D АНИМАЦИЯ с Z ЗАВЕРШЕНА!")


if __name__ == "__main__":
    interactive_gesture_viewer("data/raw")

