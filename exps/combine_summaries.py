#!/usr/bin/env python3
"""
Скрипт для объединения всех summary.tex файлов в один компилируемый LaTeX документ.
"""

import os
import re
from pathlib import Path

# Базовая директория
BASE_DIR = Path(__file__).parent

# Определяем порядок глав и параграфов (актуальные названия папок)
CHAPTERS = [
    ("Глава_1_Термодинамика_равновесных_систем", "Термодинамика равновесных систем", [
        "§01_Основные_понятия_и_первое_начало_термодинамики",
        "§02_Второе_начало_термодинамики_Энтропия",
        "§03_Термодинамические_потенциалы",
        "§04_Реальные_газы",
        "§05_Системы_с_переменным_числом_частиц",
    ]),
    ("Глава_2_Статистическая_физика_равновесных_систем", "Статистическая физика равновесных систем", [
        "§06_Некоторые_вероятностные_представления",
        "§07_Распределения_Гиббса",
        "§08_Распределение_Максвелла",
        "§09_Распределение_Больцмана",
        "§10_Цепочка_уравнений_для_равновесных_функций_распределения",
        "§11_Идеальные_квантовые_газы_в_равновесном_состоянии",
        "§12_Флуктуации_в_равновесных_системах",
    ]),
    ("Глава_3_Элементы_теории_неравновесных_процессов", "Элементы теории неравновесных процессов", [
        "§13_Кинетическое_уравнение_Больцмана_и_процессы_переноса",
        "§14_Броуновское_движение",
    ]),
]

def get_paragraph_title(folder_name):
    """Извлекает название параграфа из имени папки: §01_Название -> §1. Название"""
    match = re.match(r'§(\d+)_(.+)', folder_name)
    if match:
        num = int(match.group(1))  # убираем ведущий ноль
        title = match.group(2).replace('_', ' ')
        return f"\\S {num}. {title}"
    return folder_name

def extract_document_body(tex_content):
    """Извлекает содержимое между \\begin{document} и \\end{document}"""
    # Убираем \maketitle если есть
    tex_content = re.sub(r'\\maketitle\s*', '', tex_content)

    # Ищем содержимое документа
    match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', tex_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return tex_content

def downgrade_sections(content):
    """Понижает уровень секций: section->subsection, subsection->subsubsection"""
    # Заменяем в правильном порядке (сначала subsection, потом section)
    content = re.sub(r'\\subsection\{', r'\\subsubsection{', content)
    content = re.sub(r'\\section\{', r'\\subsection{', content)
    return content

def main():
    output_lines = []

    # Преамбула
    output_lines.append(r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian]{babel}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{geometry}
\geometry{margin=2cm}

\title{Сборник формул по статистической физике и термодинамике}
\author{Конспект формул}
\date{}

\begin{document}
\maketitle
\tableofcontents
\newpage
""")

    # Обрабатываем каждую главу
    for chapter_folder, chapter_title, paragraphs in CHAPTERS:
        chapter_path = BASE_DIR / chapter_folder

        if not chapter_path.exists():
            print(f"Папка главы не найдена: {chapter_path}")
            continue

        # Добавляем заголовок главы
        output_lines.append(f"\n\\part{{{chapter_title}}}\n")

        # Обрабатываем каждый параграф
        for para_folder in paragraphs:
            para_path = chapter_path / para_folder / "text" / "summary.tex"

            if not para_path.exists():
                print(f"Файл не найден: {para_path}")
                continue

            print(f"Обрабатываю: {para_folder}")

            # Получаем название параграфа для заголовка section
            para_title = get_paragraph_title(para_folder)

            with open(para_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Извлекаем тело документа
            body = extract_document_body(content)

            # Понижаем уровни секций
            body = downgrade_sections(body)

            # Добавляем разделитель и заголовок параграфа
            output_lines.append(f"\n%{'='*70}")
            output_lines.append(f"% {para_folder}")
            output_lines.append(f"%{'='*70}\n")

            # Добавляем section с названием параграфа
            output_lines.append(f"\\section{{{para_title}}}\n")

            # Добавляем содержимое
            output_lines.append(body)
            output_lines.append("\n\\newpage\n")

    # Закрываем документ
    output_lines.append(r"\end{document}")

    # Записываем результат
    output_path = BASE_DIR / "full_summary_combined.tex"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"\nГотово! Файл сохранён: {output_path}")

if __name__ == "__main__":
    main()
