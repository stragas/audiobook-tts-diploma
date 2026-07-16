# test_silero.py
import torch
import soundfile as sf
import numpy as np

print("🔄 Загрузка модели Silero...")

model, example_text = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='aidar_v2',
    trust_repo=True
)

print(f"🎤 Спикер: aidar_v2")
print(f"📝 Текст: {example_text}")
print("🔊 Генерация аудио...")

# Синтез речи
raw_audio = model.apply_tts(example_text, sample_rate=16000)

# ✅ Конвертация: list → Tensor → numpy
if isinstance(raw_audio, list):
    audio = raw_audio[0]  # Берём первый элемент списка
else:
    audio = raw_audio

if isinstance(audio, torch.Tensor):
    audio = audio.detach().cpu().numpy()

# ✅ КРИТИЧЕСКИ ВАЖНО: убираем размерность батча (squeeze)
# Было: (1, 64800) → Стало: (64800,)
if len(audio.shape) == 2:
    audio = audio.squeeze(0)
    print(f"📊 Убрана размерность батча: {audio.shape}")

# ✅ Проверка: должна быть 1D
if len(audio.shape) != 1:
    audio = audio.flatten()
    print(f"📊 Массив сплющен до 1D: {audio.shape}")

# ✅ Нормализация (диапазон -1..1 для WAV)
audio_max = np.max(np.abs(audio))
if audio_max > 1.0:
    audio = audio / audio_max
    print(f"📊 Нормализация: макс. амплитуда = {audio_max:.3f}")

# ✅ Sample Rate (экспериментально подтверждено)
sample_rate = 16000

# Сохранение
output_path = "test_output.wav"
sf.write(output_path, audio, sample_rate)

print(f"\n✅ Готово! Файл сохранён: {output_path}")
print(f"📊 Sample Rate: {sample_rate} Hz")
print(f"📏 Длительность: {len(audio)/sample_rate:.2f} сек")
print(f"📈 Формат: {audio.dtype}, Shape: {audio.shape}")
print(f"📊 Диапазон: [{audio.min():.3f}, {audio.max():.3f}]")