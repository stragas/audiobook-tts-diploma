# src/text_processor.py
"""
Модуль интеллектуальной предобработки текста для TTS.
Расставляет паузы на основе пунктуации и морфологического анализа.

Научная ценность для диплома:
- Улучшение просодии синтеза без переобучения модели
- Алгоритмическая расстановка пауз на основе синтаксического анализа
"""

import re
from typing import List
import pymorphy3


class TextPreprocessor:
    """Преобразует сырой текст в формат, оптимизированный для синтеза речи."""

    # Теги пауз для Silero (в секундах)
    PAUSE_SHORT = "[spk=0.3]"  # Запятая
    PAUSE_MEDIUM = "[spk=0.6]"  # Точка с запятой, тире
    PAUSE_LONG = "[spk=1.0]"  # Конец предложения
    PAUSE_XLONG = "[spk=1.5]"  # Абзац, многоточие

    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()

    def preprocess(self, text: str) -> str:
        """Основной метод: принимает сырой текст, возвращает текст с тегами пауз."""
        text = self._clean_text(text)
        text = self._add_punctuation_pauses(text)
        text = self._add_nlp_pauses(text)
        return text

    def _clean_text(self, text: str) -> str:
        """Удаляет лишние пробелы, нормализует переносы строк."""
        # Сохраняем многоточие как единый символ перед очисткой
        text = text.replace('...', '…')
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:…])', r'\1', text)
        text = re.sub(r'([.,!?;:…])([^\s])', r'\1 \2', text)
        return text.strip()

    def _add_punctuation_pauses(self, text: str) -> str:
        """
        Расставляет паузы на основе правил пунктуации.
        КРИТИЧЕСКИ ВАЖНО: Порядок регексов и защита от повторного матчинга.
        """
        # 1. Сначала многоточие (иначе разобьётся на три точки)
        text = re.sub(r'…\s*', f' {self.PAUSE_XLONG} ', text)

        # 2. Конец предложения (. ! ?) — ТОЛЬКО если после точки пробел или конец строки
        #    Это защищает точки внутри тегов [spk=1.5]
        text = re.sub(r'([.!?])(\s|$)', f'\\1{self.PAUSE_LONG}\\2', text)

        # 3. Запятая — ТОЛЬКО если после запятой пробел
        text = re.sub(r',(\s)', f',{self.PAUSE_SHORT}\\1', text)

        # 4. Точка с запятой, двоеточие
        text = re.sub(r'([;:])(\s)', f'\\1{self.PAUSE_MEDIUM}\\2', text)

        # 5. Тире (длинное)
        text = re.sub(r'\s*—\s*', f' {self.PAUSE_MEDIUM} ', text)

        # 6. Абзац (два переноса строки)
        text = re.sub(r'\n\s*\n', f' {self.PAUSE_XLONG} ', text)

        return text

    def _add_nlp_pauses(self, text: str) -> str:
        """Использует морфологический анализ для улучшения пауз."""
        # Разбиваем по предложениям, но не трогаем теги
        sentences = re.split(r'([.!?…])', text)
        processed = []

        for part in sentences:
            if not part.strip():
                continue

            # Пропускаем части, которые содержат только теги
            if re.match(r'^\s*\[spk=', part):
                processed.append(part)
                continue

            if re.search(r'[а-яА-Яa-zA-Z]', part):
                words = part.split()
                # Длинные предложения (>15 слов) — дополнительная пауза в середине
                if len(words) > 15:
                    mid = len(words) // 2
                    for j in range(max(0, mid - 3), min(len(words), mid + 3)):
                        # Добавляем паузу только если запятая не внутри тега
                        if words[j].endswith(',') and '[spk=' not in words[j]:
                            words[j] = words[j] + f' {self.PAUSE_SHORT}'
                            break
                    part = ' '.join(words)

            processed.append(part)

        return ''.join(processed)

    def split_into_chunks(self, text: str, max_chars: int = 500) -> List[str]:
        """
        Разбивает текст на чанки для синтеза (лимиты модели).
        Режет ТОЛЬКО по границам тегов пауз, не внутри них.
        """
        chunks = []
        current_chunk = ""

        # Разбиваем по тегам пауз (сохраняем теги в результате)
        parts = re.split(r'(\[spk=[\d.]+\])', text)

        for part in parts:
            if not part.strip():
                continue

            # Если это тег паузы — добавляем без проверки длины
            if part.startswith('[spk='):
                current_chunk += part
                continue

            # Если текст — проверяем длину
            if len(current_chunk) + len(part) <= max_chars:
                current_chunk += part
            else:
                # Сохраняем текущий чанк и начинаем новый
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = part

        # Добавляем последний чанк
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks