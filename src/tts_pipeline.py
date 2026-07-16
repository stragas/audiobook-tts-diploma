# src/tts_pipeline.py
"""
Интеграционный пайплайн для озвучивания аудиокниг.
Соединяет NLP-препроцессинг + TTS-синтез + пост-обработку.

Научная ценность для диплома:
- Многоступенчатая обработка текста перед синтезом
- Алгоритмическое улучшение просодии без переобучения модели
- Модульная архитектура для масштабирования
"""

import torch
import soundfile as sf
import numpy as np
import sys
from pathlib import Path

# Добавляем src в путь импорта
sys.path.append(str(Path(__file__).parent))

from text_processor import TextPreprocessor


class AudiobookPipeline:
    """
    Полный пайплайн синтеза аудиокниги:
    1. Предобработка текста (NLP)
    2. Синтез речи (TTS)
    3. Пост-обработка аудио
    """

    def __init__(self, speaker='aidar_v2', sample_rate=16000):
        """
        Инициализация пайплайна.

        Args:
            speaker: Имя спикера Silero (aidar_v2, kseniya_v2, и т.д.)
            sample_rate: Частота дискретизации (экспериментально: 16000)
        """
        self.sample_rate = sample_rate
        self.speaker = speaker
        self.preprocessor = TextPreprocessor()

        # Загрузка TTS модели
        print("🔄 Загрузка модели Silero TTS...")
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker=speaker,
            trust_repo=True
        )
        print(f"✅ Модель загружена: {speaker}")

    def synthesize(self, text: str, output_path: str = "output.wav"):
        """
        Полный цикл синтеза: текст → аудиофайл.

        Args:
            text: Исходный текст для озвучки
            output_path: Путь для сохранения аудио
        """
        print(f"\n📝 Исходный текст: {len(text)} симв.")

        # Шаг 1: Предобработка текста (NLP)
        print("🔧 Предобработка текста (NLP)...")
        processed_text = self.preprocessor.preprocess(text)
        print(f"✅ Текст с тегами пауз: {len(processed_text)} симв.")

        # Шаг 2: Разбивка на чанки (лимиты модели)
        print("✂️ Разбивка на чанки...")
        chunks = self.preprocessor.split_into_chunks(processed_text, max_chars=100)
        print(f"✅ Чанков: {len(chunks)}")

        # Шаг 3: Синтез каждого чанка
        print("🔊 Синтез речи...")
        audio_chunks = []
        for i, chunk in enumerate(chunks, 1):
            print(f"   Чанк {i}/{len(chunks)}...")
            audio = self.model.apply_tts(chunk, sample_rate=self.sample_rate)

            # Конвертация в numpy
            if isinstance(audio, list):
                audio = audio[0]
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            if len(audio.shape) == 2:
                audio = audio.squeeze(0)

            audio_chunks.append(audio)

        # Шаг 4: Конкатенация чанков
        print("🔗 Объединение чанков...")
        full_audio = np.concatenate(audio_chunks)

        # Шаг 5: Нормализация
        audio_max = np.max(np.abs(full_audio))
        if audio_max > 1.0:
            full_audio = full_audio / audio_max
            print(f"📊 Нормализация: макс. амплитуда = {audio_max:.3f}")

        # Шаг 6: Сохранение
        print(f"💾 Сохранение: {output_path}")
        sf.write(output_path, full_audio, self.sample_rate)

        duration = len(full_audio) / self.sample_rate
        print(f"\n✅ Готово!")
        print(f"📏 Длительность: {duration:.2f} сек")
        print(f"📊 Формат: {full_audio.dtype}, Shape: {full_audio.shape}")

        return output_path, duration

    def synthesize_file(self, input_path: str, output_path: str = "output.wav"):
        """
        Озвучка текста из файла.

        Args:
            input_path: Путь к текстовому файлу
            output_path: Путь для сохранения аудио
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.synthesize(text, output_path)


# Тест пайплайна
if __name__ == "__main__":
    # Инициализация
    pipeline = AudiobookPipeline(speaker='aidar_v2')

    # Тестовый текст
    test_text = """
    Было раннее утро, солнце только начинало подниматься над горизонтом. 
    Лес ещё спал, окутанный лёгким туманом; птицы молчали. 
    Вдруг — где-то вдалеке — послышался странный звук… 
    Что это было? Никто не знал.
    """

    # Запуск синтеза
    pipeline.synthesize(test_text, output_path="pipeline_test.wav")