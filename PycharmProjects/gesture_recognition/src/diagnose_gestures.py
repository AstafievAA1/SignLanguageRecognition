import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import sys

sys.path.append('..')
from model import GestureLSTM
from utils import load_config, get_device


def analyze_dataset(config_path="../configs/config.yaml"):
    print("\n" + "=" * 80)
    print(" АНАЛИЗ ДАТАСЕТА")
    print("=" * 80)

    from utils import load_config
    config = load_config(config_path)

    # Загружаем датасет
    dataset_path = os.path.join(config['data']['processed_path'],
                                config['data']['dataset_name'])

    if not os.path.exists(dataset_path):
        print(f"❌ Датасет не найден: {dataset_path}")
        print("Запусти: cd src && python data_preprocessing.py")
        return None

    data = np.load(dataset_path, allow_pickle=True)

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    actions = config['gestures']['actions']

    print(f"\n📦 Размеры датасета:")
    print(f"   Train: {len(X_train)} примеров")
    print(f"   Val:   {len(X_val)} примеров")
    print(f"   Test:  {len(X_test)} примеров")
    print(f"   ВСЕГО: {len(X_train) + len(X_val) + len(X_test)} примеров")

    # Распределение по классам
    print(f"\n📊 Распределение по классам (Train):")
    train_counts = Counter(y_train)

    for i, action in enumerate(actions):
        count = train_counts.get(i, 0)
        percentage = count / len(y_train) * 100
        bar = "█" * int(percentage / 2)
        print(f"   {action:15} : {count:4} ({percentage:5.1f}%) {bar}")

    # Проверка на несбалансированность
    counts = list(train_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    print(f"\n⚠️  Баланс датасета:")
    print(f"   Минимум примеров: {min_count}")
    print(f"   Максимум примеров: {max_count}")
    print(f"   Ratio: {imbalance_ratio:.2f}x")

    if imbalance_ratio > 2.0:
        print(f"   ❌ ПРОБЛЕМА: Датасет несбалансирован! (ratio > 2.0)")
        print(f"   Рекомендация: Собери больше данных для классов с малым количеством")
    else:
        print(f"   ✅ Датасет сбалансирован")

    # Находим проблемные классы
    avg_count = np.mean(counts)
    print(f"\n🎯 Классы с недостатком данных (< среднего {avg_count:.0f}):")
    for i, action in enumerate(actions):
        count = train_counts.get(i, 0)
        if count < avg_count:
            print(f"   ❌ {action:15} : {count:4} (нужно еще {int(avg_count - count)})")

    return {
        'train_counts': train_counts,
        'actions': actions,
        'imbalance_ratio': imbalance_ratio
    }


def analyze_confusion_matrix(config_path="../configs/config.yaml"):
    """
    Детальный анализ Confusion Matrix
    """
    print("\n" + "=" * 80)
    print("🔍 АНАЛИЗ CONFUSION MATRIX")
    print("=" * 80)

    config = load_config(config_path)
    device = get_device()

    # Находим последнюю модель
    models_path = config['data']['models_path']
    models = [f for f in os.listdir(models_path) if f.endswith('.pth')]

    if not models:
        print("❌ Модель не найдена!")
        return None

    models_with_time = [(f, os.path.getmtime(os.path.join(models_path, f))) for f in models]
    models_with_time.sort(key=lambda x: x[1], reverse=True)
    model_path = os.path.join(models_path, models_with_time[0][0])

    print(f"\n📦 Модель: {models_with_time[0][0]}")

    # Загружаем модель
    checkpoint = torch.load(model_path, map_location=device)
    model = GestureLSTM(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_classes=len(config['gestures']['actions']),
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional'],
        use_attention=config['model'].get('use_attention', False)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Загружаем test set
    dataset_path = os.path.join(config['data']['processed_path'],
                                config['data']['dataset_name'])
    data = np.load(dataset_path, allow_pickle=True)
    X_test = data['X_test']
    y_test = data['y_test']

    # Предсказания
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(len(X_test)):
            X = torch.FloatTensor(X_test[i]).unsqueeze(0).to(device)
            y = y_test[i]

            output = model(X)
            pred = torch.argmax(output, dim=1).item()

            all_preds.append(pred)
            all_labels.append(y)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    actions = config['gestures']['actions']

    # Визуализация
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=actions, yticklabels=actions)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    output_path = os.path.join(config['data']['models_path'], 'confusion_matrix.png')
    plt.savefig(output_path, dpi=150)
    print(f"\n💾 Confusion Matrix сохранена: {output_path}")
    plt.close()

    # Анализ ошибок
    print(f"\n🔍 Анализ ошибок:")
    print(f"   Всего примеров: {len(all_labels)}")
    print(f"   Правильно: {np.sum(np.array(all_preds) == np.array(all_labels))}")
    print(f"   Неправильно: {np.sum(np.array(all_preds) != np.array(all_labels))}")

    # Per-class accuracy
    print(f"\n📊 Точность по классам:")
    for i, action in enumerate(actions):
        mask = np.array(all_labels) == i
        if np.sum(mask) > 0:
            correct = np.sum((np.array(all_preds)[mask]) == i)
            total = np.sum(mask)
            acc = correct / total

            status = "✅" if acc > 0.85 else ("⚠️" if acc > 0.70 else "❌")
            print(f"   {status} {action:15} : {acc:.2%} ({correct}/{total})")

    # Находим самые частые ошибки
    print(f"\n❌ ТОП-5 самых частых ошибок:")
    errors = []
    for i in range(len(actions)):
        for j in range(len(actions)):
            if i != j and cm[i][j] > 0:
                errors.append((actions[i], actions[j], cm[i][j]))

    errors.sort(key=lambda x: x[2], reverse=True)

    for true_label, pred_label, count in errors[:5]:
        print(f"   {true_label:15} → {pred_label:15} : {count} раз")

    return {
        'confusion_matrix': cm,
        'actions': actions,
        'predictions': all_preds,
        'labels': all_labels
    }


def analyze_specific_gestures(target_gestures, config_path="../configs/config.yaml"):
    """
    Детальный анализ конкретных жестов (stop, go_forward)
    """
    print("\n" + "=" * 80)
    print(f"🎯 ДЕТАЛЬНЫЙ АНАЛИЗ ЖЕСТОВ: {', '.join(target_gestures)}")
    print("=" * 80)

    config = load_config(config_path)
    actions = config['gestures']['actions']

    # Проверяем наличие данных
    data_path = config['data']['raw_path']

    for gesture in target_gestures:
        if gesture not in actions:
            print(f"❌ Жест '{gesture}' не найден в actions!")
            continue

        gesture_path = os.path.join(data_path, gesture)

        if not os.path.exists(gesture_path):
            print(f"\n❌ {gesture}:")
            print(f"   Папка не найдена: {gesture_path}")
            print(f"   Рекомендация: Собери данные для этого жеста!")
            continue

        # Считаем последовательности
        sequences = [d for d in os.listdir(gesture_path)
                     if os.path.isdir(os.path.join(gesture_path, d))]

        print(f"\n{'✅' if len(sequences) >= 30 else '❌'} {gesture}:")
        print(f"   Последовательностей: {len(sequences)}")

        if len(sequences) < 30:
            print(f"   ⚠️  ПРОБЛЕМА: Недостаточно данных!")
            print(f"   Рекомендация: Собери минимум 30 последовательностей (есть {len(sequences)})")

        # Проверяем качество данных
        incomplete = []
        for seq in sequences:
            seq_path = os.path.join(gesture_path, seq)
            frames = [f for f in os.listdir(seq_path) if f.endswith('.npy')]
            if len(frames) != config['collection']['sequence_length']:
                incomplete.append((seq, len(frames)))

        if incomplete:
            print(f"   ⚠️  Неполных последовательностей: {len(incomplete)}")
            for seq, count in incomplete[:3]:
                print(f"      Папка {seq}: {count} фреймов")


def check_gesture_similarity(config_path="../configs/config.yaml"):
    """
    Проверка похожести жестов
    """
    print("\n" + "=" * 80)
    print("🔬 АНАЛИЗ ПОХОЖЕСТИ ЖЕСТОВ")
    print("=" * 80)

    config = load_config(config_path)
    dataset_path = os.path.join(config['data']['processed_path'],
                                config['data']['dataset_name'])

    if not os.path.exists(dataset_path):
        print("❌ Датасет не найден!")
        return

    data = np.load(dataset_path, allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    actions = config['gestures']['actions']

    # Вычисляем средний жест для каждого класса
    class_means = {}
    for i, action in enumerate(actions):
        mask = y_train == i
        if np.sum(mask) > 0:
            class_means[action] = np.mean(X_train[mask], axis=0)

    # Вычисляем расстояния между средними жестами
    print("\n📏 Расстояния между жестами (чем меньше, тем похожее):")

    similarities = []
    for i, action1 in enumerate(actions):
        if action1 not in class_means:
            continue
        for j, action2 in enumerate(actions):
            if i >= j or action2 not in class_means:
                continue

            mean1 = class_means[action1]
            mean2 = class_means[action2]

            # Евклидово расстояние
            distance = np.linalg.norm(mean1 - mean2)
            similarities.append((action1, action2, distance))

    # Сортируем по возрастанию (самые похожие первыми)
    similarities.sort(key=lambda x: x[2])

    for action1, action2, dist in similarities[:10]:
        status = "❌" if dist < 5.0 else ("⚠️" if dist < 10.0 else "✅")
        print(f"   {status} {action1:15} ↔ {action2:15} : {dist:.2f}")

    if similarities[0][2] < 5.0:
        print(f"\n❌ ПРОБЛЕМА: Жесты '{similarities[0][0]}' и '{similarities[0][1]}' "
              f"слишком похожи (distance={similarities[0][2]:.2f})")
        print(f"   Рекомендация: Сделай жесты более различимыми или собери больше данных")


def generate_recommendations(config_path="../configs/config.yaml"):
    """
    Генерация рекомендаций по улучшению
    """
    print("\n" + "=" * 80)
    print("💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ")
    print("=" * 80)

    # Анализируем все проблемы
    dataset_info = analyze_dataset(config_path)

    if dataset_info and dataset_info['imbalance_ratio'] > 2.0:
        print("\n СБАЛАНСИРУЙ ДАТАСЕТ")
        print("   Проблема: Несбалансированный датасет")
        print("   Решение:")
        print("   - Собери больше данных для классов с малым количеством")
        print("   - Используй class_weight в loss function")
        print("   - Увеличь аугментацию для малых классов")

    print("\n2️⃣ УЛУЧШИ РАЗЛИЧИМОСТЬ ЖЕСТОВ")
    print("   Проблема: Жесты 'stop' и 'go_forward' могут быть похожи")
    print("   Решение:")
    print("   - Сделай жесты более контрастными")
    print("   - 'stop': две руки вверх ладонями вперед")
    print("   - 'go_forward': две руки вытянуты вперед")
    print("   - Добавь динамическое движение (например, 'stop' = резкая остановка)")

    print("\n3️⃣ СОБЕРИ БОЛЬШЕ ДАННЫХ")
    print("   Текущее: num_sequences = 50")
    print("   Рекомендуется:")
    print("   - Минимум: 50 последовательностей")
    print("   - Хорошо: 100 последовательностей")
    print("   - Отлично: 200+ последовательностей")

    print("\n4️⃣ УЛУЧШИ АУГМЕНТАЦИЮ")
    print("   Добавь в data_preprocessing.py:")
    print("   - Временное растяжение/сжатие")
    print("   - Больше шума (0.03 вместо 0.02)")
    print("   - Rotation augmentation")
    print("   - Speed variation")

    print("\n5️⃣ ИСПОЛЬЗУЙ CLASS WEIGHTS")
    print("   В train.py замени:")
    print("   criterion = nn.CrossEntropyLoss()")
    print("   на:")
    print("   weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)")
    print("   criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))")

    print("\n6️⃣ ПЕРЕОБУЧИ МОДЕЛЬ")
    print("   После сбора данных:")
    print("   cd src")
    print("   python data_preprocessing.py")
    print("   python train.py")


def run_full_diagnostics(target_gestures=['stop', 'go_forward'],
                         config_path="../configs/config.yaml"):
    """
    Полная диагностика
    """
    print("\n" + "=" * 80)
    print("🔍 ПОЛНАЯ ДИАГНОСТИКА ПРОБЛЕМ С ЖЕСТАМИ")
    print("=" * 80)
    print(f"Анализируем жесты: {', '.join(target_gestures)}")

    # 1. Анализ датасета
    analyze_dataset(config_path)

    # 2. Анализ конкретных жестов
    analyze_specific_gestures(target_gestures, config_path)

    # 3. Confusion Matrix
    analyze_confusion_matrix(config_path)

    # 4. Похожесть жестов
    check_gesture_similarity(config_path)

    # 5. Рекомендации
    generate_recommendations(config_path)

    print("\n" + "=" * 80)
    print("✅ ДИАГНОСТИКА ЗАВЕРШЕНА")
    print("=" * 80)
    print("\nСледующие шаги:")
    print("1. Изучи Confusion Matrix (confusion_matrix.png)")
    print("2. Собери больше данных для проблемных жестов")
    print("3. Сделай жесты более различимыми")
    print("4. Переобучи модель")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Запуск полной диагностики
    run_full_diagnostics(
        target_gestures=['stop', 'go_forward'],
        config_path="../configs/config.yaml"
    )