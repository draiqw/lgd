#!/bin/bash
# Скрипт для запуска TensorBoard

echo "Запуск TensorBoard для просмотра графиков обучения..."
echo ""
echo "После запуска откройте в браузере: http://localhost:6006"
echo ""
echo "Для просмотра GA: выберите runs/agnews/ga/tensorboard"
echo "Для просмотра ES: выберите runs/agnews/es/tensorboard"
echo ""
echo "Нажмите Ctrl+C для остановки"
echo ""

# Запуск TensorBoard
poetry run tensorboard --logdir=runs/agnews --port=6006