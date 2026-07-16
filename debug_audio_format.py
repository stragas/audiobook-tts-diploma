# debug_audio_format.py
import torch
import numpy as np

print("🔍 Глубокая диагностика формата аудио от Silero v5...")

model, example_text = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='aidar_v2',
    trust_repo=True
)

# Вызываем метод
raw_audio = model.apply_tts(example_text, sample_rate=16000)

print(f"\n📋 Тип возврата: {type(raw_audio)}")
print(f"📋 Значение: {raw_audio}")

# Если это список — проверяем вложенность
if isinstance(raw_audio, list):
    print(f"\n📊 Это список длиной: {len(raw_audio)}")
    if len(raw_audio) > 0:
        print(f"   Тип первого элемента: {type(raw_audio[0])}")
        if isinstance(raw_audio[0], list):
            print(f"   ⚠️ Вложенный список! Длина вложенного: {len(raw_audio[0])}")
            print(f"   Тип вложенного элемента: {type(raw_audio[0][0])}")
            # Пробуем конвертировать вложенный список
            audio = np.array(raw_audio[0], dtype=np.float32)
        else:
            audio = np.array(raw_audio, dtype=np.float32)
    else:
        print("   ❌ Пустой список!")
        audio = None

elif isinstance(raw_audio, torch.Tensor):
    print(f"\n📊 Это Tensor")
    print(f"   Shape: {raw_audio.shape}")
    print(f"   Dtype: {raw_audio.dtype}")
    audio = raw_audio.detach().cpu().numpy()

elif isinstance(raw_audio, tuple):
    print(f"\n📊 Это кортеж длиной: {len(raw_audio)}")
    for i, item in enumerate(raw_audio):
        print(f"   Элемент {i}: {type(item)}")
    # Берём первый элемент (аудио)
    if len(raw_audio) > 0:
        if isinstance(raw_audio[0], torch.Tensor):
            audio = raw_audio[0].detach().cpu().numpy()
        elif isinstance(raw_audio[0], list):
            audio = np.array(raw_audio[0], dtype=np.float32)
        else:
            audio = np.array(raw_audio[0], dtype=np.float32)
    else:
        audio = None

elif isinstance(raw_audio, np.ndarray):
    print(f"\n📊 Это уже numpy.ndarray")
    print(f"   Shape: {raw_audio.shape}")
    print(f"   Dtype: {raw_audio.dtype}")
    audio = raw_audio

else:
    print(f"\n❌ Неизвестный тип: {type(raw_audio)}")
    audio = None

# Финальная проверка
if audio is not None:
    print(f"\n✅ Финальный массив:")
    print(f"   Type: {type(audio)}")
    print(f"   Shape: {audio.shape}")
    print(f"   Dtype: {audio.dtype}")
    print(f"   Min: {audio.min():.4f}, Max: {audio.max():.4f}")
    print(f"   Длина (сэмплов): {len(audio)}")
    print(f"   Ожидаемая длительность при 16000 Гц: {len(audio) / 16000:.2f} сек")
else:
    print("\n❌ Не удалось получить аудио-данные")