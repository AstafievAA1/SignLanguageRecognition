import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import time
from datetime import datetime
from typing import Tuple
import matplotlib.pyplot as plt

from model import GestureLSTM, print_model_summary
from data_preprocessing import prepare_dataset, create_dataloaders
from utils import load_config, get_device, print_training_info


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(train_loader), correct / total


def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(val_loader), correct / total


def train_model(config_path: str = "../configs/config.yaml"):
    config = load_config(config_path)

    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ МОДЕЛИ РАСПОЗНАВАНИЯ ЖЕСТОВ")
    print("=" * 70)

    device = get_device()

    print("\n Подготовка данных...")
    data_dict = prepare_dataset(config_path, augment=True)

    if data_dict is None:
        print(" Ошибка подготовки данных")
        return

    batch_size = config['training']['batch_size']
    train_loader, val_loader, test_loader = create_dataloaders(data_dict, batch_size)

    print(f"\n Данные готовы:")
    print(f"  Train: {len(train_loader.dataset)} примеров")
    print(f"  Val:   {len(val_loader.dataset)} примеров")
    print(f"  Test:  {len(test_loader.dataset)} примеров")

    print("\n Создание модели...")
    model = GestureLSTM(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_classes=len(config['gestures']['actions']),
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional'],
        use_attention=config['model'].get('use_attention', True)
    ).to(device)

    print_model_summary(model)

    # Label smoothing помогает при небольших датасетах
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        amsgrad=True
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['reduce_lr_factor'],
        patience=config['training']['reduce_lr_patience'],
    )

    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience']
    )

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'lr': []
    }

    print("\n→ Обучение модели...")
    print("=" * 70)

    num_epochs = config['training']['epochs']
    best_val_acc = 0.0
    best_model_path = None
    prev_lr = optimizer.param_groups[0]['lr']

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']


        if current_lr < prev_lr:
            print(f"  LR снижен: {prev_lr:.6f} → {current_lr:.6f}")
            prev_lr = current_lr

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start
        print_training_info(epoch, train_loss, val_loss, train_acc, val_acc, current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"gesture_lstm_acc{val_acc:.4f}_{timestamp}.pth"
            model_path = os.path.join(config['data']['models_path'], model_name)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, model_path)

            best_model_path = model_path
            print(f"  Лучшая модель сохранена: {model_name}")

        if early_stopping(val_loss):
            print(f"\n Early stopping на эпохе {epoch}")
            break

        print(f"  Время эпохи: {epoch_time:.2f}s")
        print("-" * 70)

    total_time = time.time() - start_time
    print("=" * 70)
    print(f" Обучение завершено за {total_time / 60:.2f} минут")
    print(f" Лучшая val accuracy: {best_val_acc:.4f}")
    print(f" Лучшая модель: {best_model_path}")

    print("\n Тестирование модели...")
    print("=" * 70)

    if best_model_path:
        checkpoint = torch.load(best_model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(" Загружена лучшая модель")

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\n Результаты на тестовой выборке:")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc:.2%})")
    print("=" * 70)

    print("\n Построение графиков...")
    plot_training_history(history, config['data']['models_path'])

    return model, history


def plot_training_history(history: dict, save_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history['lr'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)

    axes[1, 1].plot(history['val_acc'], marker='o', markersize=3)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].set_title('Validation Accuracy (Detailed)')
    axes[1, 1].grid(True)

    plt.tight_layout()

    plot_path = os.path.join(
        save_path,
        f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f" Графики сохранены: {plot_path}")
    plt.close()


if __name__ == "__main__":
    model, history = train_model()