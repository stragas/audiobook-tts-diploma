# check_env.py
import torch
import sys

print(f"✅ Python: {sys.version}")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA доступен: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ Видеокарта: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ Работа будет выполняться на CPU (это нормально для старта)")

# Проверка pymorphy3
try:
    import pymorphy3
    print("✅ Pymorphy3: установлен")
except ImportError:
    print("❌ Pymorphy3: не найден")

# Проверка Silero (через torch.hub) - финальная версия
print("\n🔄 Проверка загрузки Silero TTS...")
try:
    model, example_text = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='ru',
        speaker='kseniya_v2',
        trust_repo=True
    )
    print("✅ Silero TTS: загружается корректно")
    # Атрибут .speakers убран — в новых версиях API он не доступен
    # Доступные спикеры см. в документации: https://github.com/snakers4/silero-models
except Exception as e:
    print(f"❌ Silero TTS: ошибка загрузки")
    print(f"   Детали: {e}")

# Проверка soundfile (для сохранения аудио)
try:
    import soundfile as sf
    print("✅ Soundfile: установлен")
except ImportError:
    print("❌ Soundfile: не найден")