# test_preprocessor.py
import sys
sys.path.append('src')

from text_processor import TextPreprocessor

processor = TextPreprocessor()

# Тестовый текст с многоточием
raw_text = """
Было раннее утро, солнце только начинало подниматься над горизонтом. 
Лес ещё спал, окутанный лёгким туманом; птицы молчали. 
Вдруг — где-то вдалеке — послышался странный звук… 
Что это было? Никто не знал.
"""

print("📝 Исходный текст:")
print(raw_text)
print("\n" + "="*60 + "\n")

processed = processor.preprocess(raw_text)

print("🔊 Текст с тегами пауз:")
print(processed)
print("\n" + "="*60 + "\n")

# ✅ Проверка 1: нет ли разорванных тегов
print("🔍 Проверка целостности тегов:")
broken_tags = [t for t in processed.split() if t.startswith('[spk=') and not t.endswith(']')]
if broken_tags:
    print(f"   ❌ Найдены разорванные теги: {broken_tags}")
else:
    print(f"   ✅ Все теги целые")

# ✅ Проверка 2: нет ли вложенных тегов
print("\n🔍 Проверка на вложенность тегов:")
if '[spk=' in processed and 'spk=1.' in processed:
    # Ищем паттерн [spk=1.[spk=
    import re
    nested = re.findall(r'\[spk=[\d.]*\[spk=', processed)
    if nested:
        print(f"   ❌ Найдены вложенные теги: {nested}")
    else:
        print(f"   ✅ Вложенных тегов нет")

# ✅ Проверка 3: подсчёт тегов
tags = re.findall(r'\[spk=[\d.]+\]', processed)
print(f"\n✅ Найдено корректных тегов пауз: {len(tags)}")
print(f"   Теги: {tags}")

chunks = processor.split_into_chunks(processed, max_chars=100)
print(f"\n✂️ Разбито на {len(chunks)} чанков:")
for i, chunk in enumerate(chunks, 1):
    print(f"{i}. [{len(chunk)} симв.] {chunk[:50]}...")