#!/bin/bash

# ============================================================================
# Скрипт для автоматической установки и запуска тестов
# ============================================================================

echo "================================================================================"
echo "  УСТАНОВКА И ЗАПУСК ТЕСТОВ АЛГОРИТМОВ ОПТИМИЗАЦИИ"
echo "================================================================================"
echo ""

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функция для вывода с цветом
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}➜ $1${NC}"
}

# Шаг 1: Проверка Python
echo "Шаг 1: Проверка Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python найден: $PYTHON_VERSION"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version)
    print_success "Python найден: $PYTHON_VERSION"
    PYTHON_CMD="python"
else
    print_error "Python не найден! Установите Python 3.7+"
    exit 1
fi
echo ""

# Шаг 2: Установка зависимостей
echo "Шаг 2: Установка зависимостей..."
print_info "Устанавливаем: numpy, scipy, matplotlib, deap"
$PYTHON_CMD -m pip install numpy scipy matplotlib deap --quiet

if [ $? -eq 0 ]; then
    print_success "Зависимости установлены"
else
    print_error "Ошибка установки зависимостей"
    exit 1
fi
echo ""

# Шаг 3: Проверка импортов
echo "Шаг 3: Проверка импортов..."
$PYTHON_CMD -c "import numpy; import scipy; import matplotlib; import deap" 2>/dev/null

if [ $? -eq 0 ]; then
    print_success "Все библиотеки импортируются корректно"
else
    print_error "Ошибка импорта библиотек"
    exit 1
fi
echo ""

# Шаг 4: Запуск тестов
echo "Шаг 4: Запуск тестов..."
print_info "Запускаем test.py..."
echo ""
echo "================================================================================"
$PYTHON_CMD test.py

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    print_success "Тесты успешно завершены!"
    echo ""
    print_info "Результаты сохранены в test_results/"
    echo "  - convergence.png     : графики сходимости"
    echo "  - test_function.png   : график функции"
    echo "  - GA_test.log         : лог GA"
    echo "  - ES_test.log         : лог ES"
    echo "  - PABBO_test.log      : лог PABBO"
    echo ""
else
    echo ""
    echo "================================================================================"
    print_error "Ошибка при выполнении тестов"
    echo ""
    print_info "Попробуйте запустить вручную:"
    echo "  $PYTHON_CMD test.py"
    exit 1
fi

# Шаг 5: Проверка результатов
echo "Шаг 5: Проверка результатов..."
if [ -f "test_results/convergence.png" ] && [ -f "test_results/test_function.png" ]; then
    print_success "Графики созданы"
else
    print_error "Графики не созданы"
fi

if [ -f "test_results/GA_test.log" ] && [ -f "test_results/ES_test.log" ] && [ -f "test_results/PABBO_test.log" ]; then
    print_success "Логи созданы"
else
    print_error "Логи не созданы"
fi
echo ""

# Шаг 6: Проверка одинаковой инициализации
echo "Шаг 6: Проверка одинаковой инициализации..."
INIT_VALUES=$(grep "Initial best: T=" test_results/*.log | tail -3)
UNIQUE_VALUES=$(echo "$INIT_VALUES" | awk -F'T=' '{print $2}' | sort -u | wc -l)

if [ "$UNIQUE_VALUES" -eq 1 ]; then
    print_success "Все алгоритмы начинают с одинаковой инициализации"
else
    print_error "Инициализация различается!"
fi
echo ""

echo "================================================================================"
echo "  ✅ ВСЁ ГОТОВО!"
echo "================================================================================"
echo ""
echo "Откройте результаты:"
echo "  open test_results/convergence.png"
echo "  open test_results/test_function.png"
echo ""
echo "Или посмотрите логи:"
echo "  cat test_results/GA_test.log"
echo ""
echo "================================================================================"
