# gradio_app.py
"""
Веб-интерфейс для синтеза аудиокниг на базе Gradio.
Production-интеграция для диплома.

Требования диплома:
- Интеграция в production (веб-сервис)
- Демонстрация работы системы пользователям
"""

import gradio as gr
import sys
from pathlib import Path

# Добавляем src в путь импорта
sys.path.append(str(Path(__file__).parent / "src"))

from tts_pipeline import AudiobookPipeline


class AudiobookApp:
    """Веб-приложение для синтеза аудиокниг."""

    def __init__(self):
        self.pipeline = None

    def initialize_pipeline(self, speaker, sample_rate):
        """Ленивая инициализация модели (при первом запросе)."""
        if self.pipeline is None:
            self.pipeline = AudiobookPipeline(
                speaker=speaker,
                sample_rate=sample_rate
            )
        return self.pipeline

    def synthesize(self, text, speaker, sample_rate, progress=gr.Progress()):
        """
        Обработка запроса от пользователя.

        Args:
            text: Текст для озвучки
            speaker: Выбор голоса
            sample_rate: Частота дискретизации
            progress: Индикатор прогресса Gradio

        Returns:
            Путь к аудиофайлу или сообщение об ошибке
        """
        if not text or not text.strip():
            return None, "❌ Введите текст для озвучки"

        if len(text) > 10000:
            return None, "❌ Текст слишком длинный (макс. 10000 симв.)"

        try:
            # Инициализация пайплайна
            progress(0.1, desc="Загрузка модели...")
            pipeline = self.initialize_pipeline(speaker, sample_rate)

            # Синтез
            progress(0.3, desc="Предобработка текста (NLP)...")
            progress(0.5, desc="Синтез речи...")
            output_path, duration = pipeline.synthesize(
                text,
                output_path="gradio_output.wav"
            )
            progress(1.0, desc="Готово!")

            return output_path, f"✅ Аудио создано! Длительность: {duration:.2f} сек"

        except Exception as e:
            return None, f"❌ Ошибка: {str(e)}"

    def create_interface(self):
        """Создание Gradio-интерфейса."""

        with gr.Blocks(title="📚 Синтез Аудиокниг AI") as app:
            gr.Markdown("""
            # 📚 Синтез Аудиокниг на базе нейросети

            Введите текст для озвучки — система автоматически:
            1. ✅ Проанализирует текст (NLP)
            2. ✅ Расставит паузы и акценты
            3. ✅ Синтезирует речь нейросетью Silero TTS
            4. ✅ Нормализует и сохранит аудио

            **Научная основа:** Алгоритмическое улучшение просодии 
            без переобучения модели (дипломный проект).
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="📝 Текст для озвучки",
                        placeholder="Введите текст здесь...",
                        lines=10,
                        max_lines=20
                    )

                    with gr.Row():
                        speaker_dropdown = gr.Dropdown(
                            label="🎤 Голос",
                            choices=[
                                ("Айдар (мужской, глубокий)", "aidar_v2"),
                                ("Ксения (женский, нейтральный)", "kseniya_v2"),
                                ("Бая (женский, мягкий)", "baya_v2"),
                                ("Наташа (женский, энергичный)", "natasha_v2"),
                                ("Руслан (мужской, чёткий)", "ruslan_v2"),
                            ],
                            value="aidar_v2"
                        )

                        sample_rate_dropdown = gr.Dropdown(
                            label="📊 Частота (Гц)",
                            choices=[
                                ("16000 (рекомендуется)", 16000),
                                ("24000 (высокое качество)", 24000),
                            ],
                            value=16000
                        )

                    synthesize_btn = gr.Button(
                        "🔊 Озвучить текст",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    audio_output = gr.Audio(
                        label="🎧 Результат",
                        type="filepath"
                    )

                    status_output = gr.Textbox(
                        label="📋 Статус",
                        interactive=False
                    )

            # Примеры текстов
            gr.Examples(
                examples=[
                    [
                        "Было раннее утро, солнце только начинало подниматься над горизонтом. Лес ещё спал, окутанный лёгким туманом; птицы молчали."],
                    ["Вдруг — где-то вдалеке — послышался странный звук… Что это было? Никто не знал."],
                    [
                        "Научно-технический прогресс развивается стремительно. Искусственный интеллект проникает во все сферы нашей жизни!"],
                ],
                inputs=text_input
            )

            # Подключение обработчика
            synthesize_btn.click(
                fn=self.synthesize,
                inputs=[text_input, speaker_dropdown, sample_rate_dropdown],
                outputs=[audio_output, status_output]
            )

            gr.Markdown("""
            ---
            **Дипломный проект:** «Нейронная сеть для озвучивания аудиокниг»

            **Технологии:** Silero TTS v5, PyTorch, NLP (pymorphy3), Gradio

            **Научный вклад:** Алгоритмическая расстановка пауз на основе 
            синтаксического анализа текста.
            """)

        return app


# Запуск приложения
if __name__ == "__main__":
    app = AudiobookApp()
    interface = app.create_interface()
    interface.launch(share=False)  # share=True для публичной ссылки