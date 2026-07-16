# debug_sample_rate.py
import torch
import numpy as np

print("🔍 Исследование sample rate Silero v5...")

model, example_text = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='aidar_v2',
    trust_repo=True
)

# Проверяем все возможные атрибуты
print("\n📋 Доступные атрибуты модели:")
for attr in dir(model):
    if not attr.startswith('_'):
        try:
            value = getattr(model, attr)
            if not callable(value):
                print(f"  {attr}: {value}")
        except:
            pass

# Проверяем конфиг
print("\n📋 Проверка config:")
if hasattr(model, 'config'):
    print(f"  config: {model.config}")
    if hasattr(model.config, 'sample_rate'):
        print(f"  config.sample_rate: {model.config.sample_rate}")

# Генерируем тестовое аудио и проверяем длину
audio = model.apply_tts(example_text)
if isinstance(audio, list):
    audio = np.array(audio, dtype=np.float32)

print(f"\n📊 Длина аудио: {len(audio)} сэмплов")
print(f"📝 Текст: {example_text}")
print(f"📏 Ожидаемая длительность (нормальная речь): ~4-6 сек")

# Рассчитываем фактический sample rate
for sr in [48000, 24000, 16000, 8000]:
    duration = len(audio) / sr
    print(f"  При sample_rate={sr}: длительность = {duration:.2f} сек")