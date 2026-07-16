# debug_model_structure.py
import torch
import inspect

print("🔍 Глубокое исследование структуры модели Silero v5...")

model, example_text = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='aidar_v2',
    trust_repo=True
)

print(f"\n📋 Тип модели: {type(model)}")
print(f"📋 Пример текста: {example_text}")

# Исследуем все методы модели
print("\n📋 Доступные методы модели:")
for attr in dir(model):
    if not attr.startswith('_'):
        value = getattr(model, attr)
        if callable(value):
            try:
                sig = inspect.signature(value)
                print(f"  {attr}{sig}")
            except:
                print(f"  {attr}(...)")

# Пробуем разные способы вызова
print("\n🧪 Тестирование различных способов синтеза:")

# Способ 1: Прямой вызов модели
print("\n1️⃣ Прямой вызов model(text, speaker)...")
try:
    audio = model(example_text, 'aidar_v2')
    print(f"   ✅ Тип результата: {type(audio)}")
    if hasattr(audio, '__len__'):
        print(f"   ✅ Длина: {len(audio)}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

# Способ 2: apply_tts с разными параметрами
print("\n2️⃣ model.apply_tts(text, speaker)...")
try:
    audio = model.apply_tts(example_text, 'aidar_v2')
    print(f"   ✅ Тип результата: {type(audio)}")
    if hasattr(audio, '__len__'):
        print(f"   ✅ Длина: {len(audio)}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

# Способ 3: apply_tts с именованными параметрами
print("\n3️⃣ model.apply_tts(text=..., speaker=...)...")
try:
    audio = model.apply_tts(text=example_text, speaker='aidar_v2')
    print(f"   ✅ Тип результата: {type(audio)}")
    if hasattr(audio, '__len__'):
        print(f"   ✅ Длина: {len(audio)}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

# Способ 4: Проверка атрибутов sample_rate
print("\n📋 Поиск sample_rate:")
for attr in dir(model):
    if 'rate' in attr.lower() or 'sr' in attr.lower() or 'sample' in attr.lower():
        try:
            value = getattr(model, attr)
            if not callable(value):
                print(f"   {attr}: {value}")
        except:
            pass