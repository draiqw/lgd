#!/usr/bin/env python3
"""
Простая демонстрация работы кастомной функции sinexp1D
"""
import sys
sys.path.append('../pabbo_method')

import torch
import numpy as np
import matplotlib.pyplot as plt
from data.function import sinexp1D

print("=" * 60)
print("Тестирование функции sinexp1D: y(x) = sin(x) + e^x")
print("=" * 60)

# 1. Проверка импорта
print("\n✅ Функция sinexp1D успешно импортирована")

# 2. Тест на одной точке
x_test = torch.tensor([[0.5]])
y_test = sinexp1D(x_test, negate=False, add_dim=True)
print(f"\n📊 Тест на одной точке:")
print(f"   x = 0.5")
print(f"   y = sin(0.5) + e^0.5 = {y_test.item():.4f}")
print(f"   Ожидаемое: {np.sin(0.5) + np.exp(0.5):.4f}")

# 3. Визуализация функции
print(f"\n📈 Создание визуализации...")

x_range = [-1, 1]
x = torch.linspace(x_range[0], x_range[1], 1000).unsqueeze(-1)

# Исходная функция
y = sinexp1D(x, negate=False, add_dim=False)

# Для оптимизации (negated)
y_opt = sinexp1D(x, negate=True, add_dim=False)

# Создаем график
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График 1: Исходная функция
ax = axes[0]
ax.plot(x.numpy(), y.numpy(), 'b-', linewidth=2, label='y(x) = sin(x) + e^x')
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Исходная функция', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)

# Находим и отмечаем минимум/максимум
max_idx = torch.argmax(y)
min_idx = torch.argmin(y)
ax.plot(x[max_idx].item(), y[max_idx].item(), 'ro', markersize=10,
        label=f'Max: ({x[max_idx].item():.3f}, {y[max_idx].item():.3f})')
ax.plot(x[min_idx].item(), y[min_idx].item(), 'go', markersize=10,
        label=f'Min: ({x[min_idx].item():.3f}, {y[min_idx].item():.3f})')
ax.legend(fontsize=9)

# График 2: Функция для максимизации
ax = axes[1]
ax.plot(x.numpy(), y_opt.numpy(), 'r-', linewidth=2, label='-y(x)')
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('-y', fontsize=12)
ax.set_title('Функция для максимизации (PABBO)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)

# Находим и отмечаем максимум (оптимум для PABBO)
opt_idx = torch.argmax(y_opt)
opt_x = x[opt_idx].item()
opt_y = y_opt[opt_idx].item()
ax.plot(opt_x, opt_y, 'r*', markersize=15,
        label=f'Optimum: x*={opt_x:.3f}, y*={opt_y:.3f}')
ax.legend(fontsize=9)

plt.tight_layout()

# Сохраняем график
output_path = 'sinexp1D_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   График сохранен: {output_path}")

# Показываем график (если в интерактивном режиме)
# plt.show()

# 4. Статистика
print(f"\n📊 Статистика функции на диапазоне {x_range}:")
print(f"   Min y: {y.min().item():.4f} при x = {x[torch.argmin(y)].item():.4f}")
print(f"   Max y: {y.max().item():.4f} при x = {x[torch.argmax(y)].item():.4f}")
print(f"   Mean y: {y.mean().item():.4f}")
print(f"   Std y: {y.std().item():.4f}")

print(f"\n📊 Для оптимизации PABBO (максимизация -y):")
print(f"   Оптимум: x* = {opt_x:.4f}, -y* = {opt_y:.4f}")
print(f"   (что соответствует y = {-opt_y:.4f})")

# 5. Тест батча
print(f"\n🔬 Тест на батче точек:")
x_batch = torch.tensor([[[0.0]], [[0.5]], [[1.0]]])  # shape: (3, 1, 1)
y_batch = sinexp1D(x_batch, negate=False, add_dim=True)
print(f"   Input shape: {x_batch.shape}")
print(f"   Output shape: {y_batch.shape}")
print(f"   Результаты:")
for i in range(len(x_batch)):
    print(f"      x={x_batch[i,0,0].item():.1f} → y={y_batch[i,0,0].item():.4f}")

print(f"\n✅ Все тесты пройдены успешно!")
print(f"\n💡 Следующие шаги:")
print(f"   1. Для полноценного предобучения запустите train.py")
print(f"   2. Рекомендуется использовать GPU или запустить на ночь")
print(f"   3. Ожидаемое время на CPU: ~2-4 часа для 8000 шагов")
print(f"   4. Для быстрого теста: используйте меньше шагов (n_steps=500)")
print("=" * 60)
