"""Build a six-page report from the actual experiment, then render for visual QA."""

import json
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/VladimirPechenov/lab06-rl-agent"
VIDEO = "https://drive.google.com/file/d/1nLk3H5OV0YQchjVa4kzpn48OO6WsL1uk/view?usp=drivesdk"


def build():
    windows = Path("C:/Windows/Fonts/arial.ttf").exists()
    pdfmetrics.registerFont(
        TTFont("Body", "C:/Windows/Fonts/arial.ttf" if windows else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )
    pdfmetrics.registerFont(
        TTFont(
            "Bold",
            "C:/Windows/Fonts/arialbd.ttf" if windows else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        )
    )
    body = ParagraphStyle("body", fontName="Body", fontSize=11, leading=16, spaceAfter=11)
    head = ParagraphStyle("head", fontName="Bold", fontSize=17, leading=23, spaceAfter=18)
    small = ParagraphStyle("small", fontName="Body", fontSize=9, leading=13, spaceAfter=8)
    story, md = [], []

    def text(value, heading=False, compact=False):
        story.append(Paragraph(escape(value), head if heading else small if compact else body))
        md.append(("## " if heading else "") + value + "\n")

    def link(label, url):
        story.append(Paragraph(f'<link href="{escape(url)}" color="#2563eb">{escape(label)}</link>', body))
        md.append(f"[{label}]({url})\n")

    def picture(path, width, height, caption):
        story.append(Image(str(ROOT / path), width=width, height=height))
        text(caption, compact=True)
        md.append(f"![{caption}]({Path(path).relative_to('docs') if path.startswith('docs/') else '../' + path})\n")

    def page():
        story.append(PageBreak())

    ppo = json.loads((ROOT / "results/eval_ppo.json").read_text())
    dqn = json.loads((ROOT / "results/eval_dqn.json").read_text())
    meta = {a: json.loads((ROOT / f"results/{a}/train.json").read_text()) for a in ("ppo", "dqn")}

    text("Лабораторная работа № 6", heading=True)
    text("Обучение мини-RL агента в игровой среде", heading=True)
    text("Дисциплина: «Игровой искусственный интеллект»")
    text("Студент: Печенов В. В. | Группа: ИВТ-465 | Дата: 17.11.2026")
    text("Стек: Python 3.12, Gymnasium 1.2.3, Stable-Baselines3 2.7.1, PyTorch, TensorBoard.")
    story.append(Spacer(1, 14))
    text("1. Цель работы", heading=True)
    text(
        "Освоить цикл обучения с подкреплением: формализовать среду как MDP, обучить агента, "
        "сравнить два алгоритма по динамике награды и проверить поведение сохранённой политики."
    )
    text("2. Постановка задачи", heading=True)
    text(
        "Выбрана CartPole-v1. Требуется получить среднюю награду не ниже 450, построить сравнительный "
        "график двух алгоритмов и записать пять эпизодов. Для PPO и DQN выделено по 100352 шага "
        "среды. Итоговая оценка - 20 отдельных эпизодов, действия детерминированы."
    )
    text(
        "Результат: PPO = 500,00 ± 0,00; DQN = 353,75 ± 66,96. Порог выполняет PPO. "
        "Числа получены реальным обучением и пересчитываются из результатов каждого эпизода."
    )
    text(
        "Оформление выполнено по разделам предоставленного репозитория. Отдельный кафедральный "
        "макет и ФИО преподавателя в доступных материалах отсутствуют.",
        compact=True,
    )
    page()

    text("3. Описание реализации: MDP", heading=True)
    text(
        "Состояние s = (x, v, θ, ω) содержит координату тележки, её скорость, угол шеста "
        "и угловую скорость. Наблюдение - четыре числа float32; изображения для обучения не используются."
    )
    text(
        "Действия: 0 - приложить силу -10 Н, 1 - приложить +10 Н. Шаг физики равен 0,02 с. "
        "После фиксации начального состояния переходы определяются динамикой среды."
    )
    text(
        "Награда: +1 на каждом шаге, включая шаг завершения. Максимум за эпизод - 500. "
        "Дисконт γ=0,99 задаёт приоритет близких наград в обучении; в таблицах приведена "
        "недисконтированная сумма наград."
    )
    text(
        "Начальное состояние: независимые значения из U(-0,05; 0,05). Флаг terminated возникает "
        "при |x| > 2,4 м либо |θ| > 12°; truncated возникает на лимите 500 шагов. Оценка и видео "
        "завершают эпизод по любому из этих флагов. Ограничения наблюдения Box шире границ завершения."
    )
    text("Архитектура", heading=True)
    text(
        "YAML → train.py → Monitor + TensorBoard → финальная модель ZIP. Отдельные ветви: "
        "evaluate.py → результаты 20 эпизодов; plot.py → сравнительный график; record.py → пять "
        "полных эпизодов. audit.py проверяет согласованность опубликованных чисел и хешей."
    )
    text(
        "Политики MlpPolicy используют штатные сети SB3: у PPO отдельные ветви actor/critic "
        "по 64–64 нейрона с Tanh; у DQN Q-сеть 64–64 с ReLU и target network. "
        "Входы и награда не нормализуются, reward shaping отсутствует."
    )
    text(
        "Порог лабораторной 450 отличается от reward_threshold=475 в спецификации Gymnasium. "
        "PPO превышает оба порога на выбранных 20 эпизодах.",
        compact=True,
    )
    page()

    text("3. Описание реализации: обучение", heading=True)
    text(
        "PPO - on-policy actor-critic: оценивает преимущества на собранных траекториях и ограничивает "
        "изменение политики с помощью clipped objective. DQN - off-policy алгоритм: обучает Q-функцию "
        "по выборкам replay buffer, использует target network и ε-greedy исследование."
    )
    text(
        "PPO: learning_rate=0,0003; n_steps=2048; batch_size=64; gamma=0,99. "
        "Остальные параметры SB3 2.7.1: n_epochs=10, gae_lambda=0,95, clip_range=0,2."
    )
    text(
        "DQN: learning_rate=0,001; buffer_size=50000; exploration_fraction=0,2; gamma=0,99. "
        "Штатные параметры: batch_size=32, learning_starts=100, train_freq=4, gradient_steps=1, "
        "target_update_interval=10000, exploration_final_eps=0,05."
    )
    text(
        "Изменение относительно шаблона: total_timesteps=100000 заменён на 100352 у обоих алгоритмов. "
        "Это ровно 49 rollout-блоков PPO по 2048 шагов; DQN получает тот же фактический бюджет. "
        "Параметры оптимизации не подбирались по итоговым тестам. CPU, один поток PyTorch, seed=42."
    )
    text("Разделение данных", heading=True)
    text(
        "Обучение использует seed=42. На шагах 0, 10000, …, 100000 и 100352 выполняется валидация "
        "на seed=1000..1009. Контрольные точки по ней не выбираются. После обучения финальные "
        "модели проверяются на seed=2000..2019; видео записано на seed=3000..3004."
    )
    text(
        "Monitor сохраняет длину, награду и время каждого завершённого эпизода; JSON содержит "
        "конфигурацию, фактическое число шагов, версии библиотек и SHA-256 модели. Одинаковые seed "
        "позволяют сравнить политики при одинаковых стартовых состояниях."
    )
    text(
        "Пакет Box2D исключён из зависимостей, поскольку выбрана CartPole. Он не нужен для этой среды. "
        "Точные версии окружения зафиксированы в requirements-lock.txt.",
        compact=True,
    )
    page()

    text("4. Демонстрация", heading=True)
    picture("docs/img/ppo_episode.png", 450, 300, "Рисунок 1. Фактический кадр первого эпизода PPO, шаг 250.")
    text(
        "Записаны пять полных эпизодов финальной PPO-политики. Каждый длится 500 шагов и "
        "приносит награду 500. Частота - 50 кадров/с; всего 2500 кадров, 50 секунд. "
        "На видео указаны номер эпизода, seed, шаг и накопленная награда."
    )
    text(
        "Кадры получены из Gymnasium render_mode=rgb_array. Вырезания и ускорения отсутствуют; "
        "добавлены только текстовые счётчики. Звука нет. Видео хранится вне Git, как требует шаблон."
    )
    link("Видео пяти эпизодов на Google Диске", VIDEO)
    link("Резервная MP4 и модели в Release v1.0", REPO + "/releases/tag/v1.0")
    text(
        "results/video_ppo.json содержит seed каждого эпизода, длину, флаги завершения, "
        "число кадров и контрольные суммы видео и модели. Максимальная награда объясняется "
        "достижением временного лимита, а не остановкой записи до падения.",
        compact=True,
    )
    page()

    text("5. Метрики и результаты", heading=True)
    rows = [["Алгоритм", "Шаги", "Mean ± std", "Min / max", "Время, с"]]
    for algo, result in [("ppo", ppo), ("dqn", dqn)]:
        rows.append(
            [
                algo.upper(),
                "100352",
                f"{result['mean_reward']:.2f} ± {result['std_reward']:.2f}",
                f"{result['min_reward']:.0f} / {result['max_reward']:.0f}",
                f"{meta[algo]['wall_seconds_including_validation']:.1f}",
            ]
        )
    table = Table([[Paragraph(escape(c), small) for c in row] for row in rows], colWidths=[65, 65, 120, 90, 110])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7eef8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)
    md.extend(["| " + " | ".join(rows[0]) + " |", "|---|---|---|---|---|"])
    md.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    picture("results/learning_curves.png", 470, 176, "Рисунок 2. Обучение PPO и DQN при равном числе шагов среды.")
    text(
        "Слева: скользящая средняя по 100 обучающим эпизодам, ранние точки используют доступные "
        "эпизоды. Справа: средняя награда и стандартное отклонение по 10 фиксированным "
        "валидационным seed. Интервалы ±std не являются доверительными интервалами."
    )
    text(
        "PPO достигает 470,4 на валидации к 20000 шагам, затем 500 к 30000 и удерживает "
        "этот уровень до конца. У DQN есть рост до 244,3 на 70000 шагах и последующий спад. "
        "На 100352 шагах валидация DQN даёт 384,7; независимая итоговая оценка - 353,75."
    )
    text(
        "Std рассчитано с ddof=0. Время включает создание модели и периодическую валидацию; "
        "процессы обучения запускались одновременно, поэтому время не является чистым бенчмарком скорости."
    )
    page()

    text("6. Выводы", heading=True)
    text(
        "Цель достигнута: финальный PPO удержал шест до лимита во всех 20 итоговых эпизодах "
        "и превысил порог лабораторной на 50 баллов. Сохранённая модель также прошла пять "
        "новых эпизодов видеодемонстрации. DQN при тех же шагах и параметрах шаблона порог не достиг."
    )
    text(
        "Результат показывает преимущество данной PPO-конфигурации на CartPole при заданном бюджете. "
        "Он не доказывает общего превосходства PPO над DQN: на алгоритм выполнен один training seed, "
        "а оптимизация DQN и другие среды не исследованы. Повтор с несколькими seed и подбор "
        "параметров только по валидации - следующий эксперимент."
    )
    text("Проверки и воспроизведение", heading=True)
    text(
        "После установки: python -m pytest; python -m ruff check .; python -m lab06.audit. "
        "Для проверки весов распакуйте lab06_artifacts.zip из Release в корень проекта и выполните "
        "python -m lab06.audit --models, затем python -m lab06.evaluate --algo ppo."
    )
    text(
        "Чтобы повторить обучение, сначала переместите results/ppo и results/dqn в архив: "
        "train.py защищает сохранённый эксперимент от случайной перезаписи. Команды обучения "
        "приведены в README. Совпадение байтов модели на другой версии ПО не гарантируется."
    )
    text("7. Список литературы", heading=True)
    for label, url in [
        ("Задание и шаблон ЛР №6", "https://github.com/kolesnikov-vstu-gameai/lab06-rl-agent"),
        ("Gymnasium: CartPole-v1", "https://gymnasium.farama.org/environments/classic_control/cart_pole/"),
        ("Stable-Baselines3: PPO", "https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/ppo.html"),
        ("Stable-Baselines3: DQN", "https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/dqn.html"),
    ]:
        link(label, url)
    text("Приложение. Ссылка на репозиторий", heading=True)
    link(REPO, REPO)

    def footer(canvas, doc):
        canvas.setFont("Body", 9)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(50, 28, "ЛР №6 | Печенов В. В. | ИВТ-465")
        canvas.drawRightString(A4[0] - 50, 28, str(doc.page))

    SimpleDocTemplate(
        str(ROOT / "docs/report.pdf"),
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=45,
        bottomMargin=45,
        title="ЛР №6 - PPO и DQN",
        author="Печенов В. В.",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    (ROOT / "docs/report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    build()
