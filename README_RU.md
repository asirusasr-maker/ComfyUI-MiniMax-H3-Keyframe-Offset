# ComfyUI-MiniMax-H3-Keyframe-Offset

[![ComfyUI](https://img.shields.io/badge/ComfyUI-0.3+-green.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Кастомные ноды для **MiniMax H3** в ComfyUI:

- **Keyframe Offset** — размещение first/last keyframe на произвольных индексах кадров, а не только в начале/конце клипа
- **Audio Generator (AIO)** — генерация аудио из текста одной нодой с опциональным **клонированием голоса** и **визуальным референсом** через Ref2VA

---

## Содержание

- [Возможности](#возможности)
- [Установка](#установка)
- [Зависимости](#зависимости)
- [Ноды](#ноды)
  - [MiniMax H3 Keyframe Offset](#1-minimax-h3-keyframe-offset)
  - [MiniMax H3 Audio Generator (AIO)](#2-minimax-h3-audio-generator-aio)
- [Примеры workflow](#примеры-workflow)
- [Технические замечания](#технические-замечания)
- [Решение проблем](#решение-проблем)
- [Лицензия](#лицензия)
- [Благодарности](#благодарности)

---

## Возможности

| Возможность | Описание |
|-------------|----------|
| Произвольные keyframe | First/last кадры можно ставить на любой индекс внутри клипа |
| Безопасный clamp offset | Offset всегда ограничивается диапазоном `[0, frame_count-1]` по реальной длине |
| Внешний `length` | Если длину задаёт другая нода, clamp всё равно идёт по вычисленному `frame_count` |
| Обход ограничения ядра | Патч `PackedLayout` при загрузке — ненулевые индексы keyframe принимаются core |
| Аудио «всё в одном» | Одна нода: промпт → шум → сэмплинг → Audio VAE decode → ComfyUI `AUDIO` |
| Нативный путь H3 | `comfy.sample.sample_custom`, `cfg=1.0` (guidance-distilled) |
| **Клонирование голоса** | Подайте `ref_audio` — модель копирует тембр/стиль и генерирует новое аудио |
| **Визуальное аудио-рассуждение** | Подайте `ref_image` — модель интерпретирует сцену и адаптирует звук (например, эхо в горах vs. эхо в комнате) |

---

## Установка

### Способ 1: Git clone (рекомендуется)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/asirusasr-maker/ComfyUI-MiniMax-H3-Keyframe-Offset.git
```

Перезапустите ComfyUI (или **Reload Custom Nodes**).

### Способ 2: Вручную

1. Скачайте репозиторий ZIP
2. Распакуйте в `ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Keyframe-Offset`
3. Перезапустите ComfyUI

### Portable ComfyUI (Windows)

Положите папку сюда:

```
ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Keyframe-Offset
```

---

## Зависимости

Пакет использует **только API ядра ComfyUI** (`torch`, `comfy.*`, `node_helpers`).
Отдельные pip-пакеты **не требуются** (кроме уже установленного MiniMax H3 в ComfyUI).

`requirements.txt` намеренно без сторонних зависимостей:

```text
# No external dependencies beyond ComfyUI core (torch, comfy)
```

Нужны официальные веса MiniMax H3: diffusion model, CLIP/text encoder, video VAE, audio VAE.

---

## Ноды

Скриншоты: [`docs/images/`](docs/images/).

---

### 1. MiniMax H3 Keyframe Offset

<img src="docs/images/keyframe_offset.png" width="420" alt="MiniMax H3 Keyframe Offset">

Нода conditioning для MiniMax H3 image-to-video с **перемещаемыми** first/last keyframe.

В стоковых workflow обычно:

- first keyframe → кадр `0`
- last keyframe → последний кадр

Эта нода позволяет поставить каждый keyframe на **любой индекс** внутри клипа. Модель свободно генерирует движение до, между и после вставленных кадров.

#### Входы

| Имя | Тип | Описание |
|-----|-----|----------|
| `clip` | CLIP | MiniMax H3 CLIP / text encoder |
| `vae` | VAE | MiniMax H3 video VAE |
| `prompt` | STRING | Текстовый промпт (dynamic prompts) |
| `width` | INT | Ширина (по умолчанию `1344`, шаг `32`) |
| `height` | INT | Высота (по умолчанию `768`, шаг `32`) |
| `length` | INT | Число кадров при 24 fps (выравнивается в сетку H3 `17k+5`; по умолчанию `124` ≈ 5 с) |
| `first_frame_offset` | INT | Индекс для `first_frame` (`0` = начало). Clamp в `[0, frame_count-1]` |
| `last_frame_offset` | INT | Индекс для `last_frame` (`-1` = последний кадр). Clamp в `[0, frame_count-1]` |
| `first_frame` | IMAGE (опц.) | Первый keyframe |
| `last_frame` | IMAGE (опц.) | Последний keyframe |

#### Выходы

| Имя | Тип | Описание |
|-----|-----|----------|
| `positive` | CONDITIONING | Conditioning с `minimax_keyframes` и `minimax_frame_count` |
| `latent` | LATENT | Пустой AV NestedTensor latent для семплинга |

#### Поведение

1. `length` выравнивается через `align_frame_count` (временная сетка H3).
2. `last_frame_offset = -1` → `frame_count - 1`.
3. Оба offset обрезаются в `[0, frame_count - 1]`.
4. Если значение изменилось, в консоль пишется сообщение:

```text
[MiniMax H3 Keyframe Offset] Offsets clamped to [0, 123]: first_frame_offset 200->123, last_frame_offset 150->123
```

5. First frame: resize с `crop="disabled"`. Last frame: `crop="center"`.
6. При загрузке патчится `PackedLayout`, чтобы ненулевые `resolved_frame_index` принимались ядром.

#### Типичный граф

```text
Load Image (first) ──┐
                     ├─► MiniMax H3 Keyframe Offset ─► KSampler / SamplerCustom ─► Decode
Load Image (last)  ──┘         ▲
                               │
                     CLIP + VAE + prompt + length
```

---

### 2. MiniMax H3 Audio Generator (AIO)

<img src="docs/images/audio_generator.png" width="420" alt="MiniMax H3 Audio Generator AIO">

Нода **текст → аудио** «всё в одном» для MiniMax H3. Отдельные conditioning / sampler / decode не нужны.

Теперь с **Ref2VA-референсами** — переключите `use_references`, чтобы открыть входы `ref_image` и `ref_audio` для расширенного контроля.

#### Входы

| Имя | Тип | Описание |
|-----|-----|----------|
| `model` | MODEL | MiniMax H3 diffusion model |
| `clip` | CLIP | MiniMax H3 CLIP |
| `vae` | VAE | MiniMax H3 **video** VAE (нужен для joint AV latent) |
| `audio_vae` | VAE | MiniMax H3 **audio** VAE (только decode) |
| `prompt` | STRING | Описание звука. С включёнными референсами обращайтесь к ним как `<Picture 1>` и `<Audio 1>` |
| `length` | INT | Длительность в кадрах при 24 fps (по умолчанию `124` ≈ 5 с) |
| `seed` | INT | Seed (`управление после генерации` / randomize) |
| `steps` | INT | Число шагов (по умолчанию `20`) |
| `sampler_name` | COMBO | Семплер (по умолчанию `res_multistep`) |
| `scheduler` | COMBO | Планировщик (по умолчанию `simple`) |
| `denoise` | FLOAT | Сила денойза `0.0–1.0` (по умолчанию `1.0`) |
| `use_references` | BOOLEAN | **Включить Ref2VA** — появляются входы `ref_image` и `ref_audio` |
| `ref_image_size` | COMBO | `match` (быстрее, масштаб под canvas) или `max` (2048 по короткой стороне, лучше identity) |
| `ref_image` | IMAGE (опц.) | Визуальный якорь → `<Picture 1>`. Помогает модели «рассуждать» об акустической среде |
| `ref_audio` | AUDIO (опц.) | Аудио-референс → `<Audio 1>`. Для клонирования голоса или переноса стиля |

#### Выходы

| Имя | Тип | Описание |
|-----|-----|----------|
| `audio` | AUDIO | Dict ComfyUI: `{"waveform": [B,C,L], "sample_rate": 32000}` |

#### Режимы референсов

Когда `use_references` **включён**, нода строит нативные `minimax_refs` (тот же путь, что и у MiniMax H3 Reference to Video).

**Клонирование голоса** (`ref_audio`)
- Подключите короткую запись голоса к `ref_audio`.
- Промпт: `A narrator reading a poem, calm and clear`
- Модель копирует тембр диктора и генерирует новую речь этим голосом.
- ⚠️ Нативный H3 предпочитает `ref_image` вместе с `ref_audio`. Без визуального якоря качество клона может упасть.

**Визуальное аудио-рассуждение** (`ref_image`)
- Подключите фото к `ref_image`.
- Промпт: `echo reverberating through the valley`
- Модель читает сцену (горы, узкая комната, собор, лес…) и адаптирует реверберацию, атмосферу и пространственный характер звука.
- Пример: один и тот же промпт `echo` + фото **гор** → широкое, далёкое эхо; + фото **маленькой комнаты** → короткое, «slap-back» эхо.

**Комбинированный** (`ref_image` + `ref_audio`)
- Используйте оба референса для максимального контроля: визуальный контекст + клонированный голос / стиль.
- Промпт: `<Picture 1> A mysterious voice whispers ancient words, <Audio 1>`

#### Внутренний пайплайн

1. Пустой AV NestedTensor latent (video placeholder `320×320` + audio latent 40 Hz).
2. Если `use_references` включён, кодируются `ref_image` (через video VAE) и/или `ref_audio` (через audio VAE) в нативные ref-блоки.
3. Токенизация промпта (с `minimax_ref_items` при активных референсах) и encode conditioning.
4. Sampler + sigmas по scheduler/steps.
5. `prepare_noise` + `comfy.sample.sample_custom`.
6. Аудио-ветка NestedTensor → `audio_vae.decode`.
7. Нормализация в ComfyUI `AUDIO`, **32 kHz stereo**.

#### CFG

Нативный MiniMax H3 — **guidance-distilled**. CFG **всегда `1.0`**, параметр в UI скрыт.

#### Типичный граф

**Базовый (только текст):**
```text
UNETLoader ──► model ─┐
CLIPLoader ─► clip  ─┤
VAELoader  ─► vae   ─┼─► MiniMax H3 Audio Generator (AIO) ─► Preview Audio / Save Audio
VAELoader  ─► audio_vae ─┘
```

**С референсами:**
```text
Load Image  ──► ref_image ──┐
Load Audio  ──► ref_audio ──┼─► MiniMax H3 Audio Generator (AIO) ─► Preview Audio
                             │
UNETLoader ──► model ────────┤
CLIPLoader ──► clip ─────────┤
VAELoader  ──► vae ──────────┤
VAELoader  ──► audio_vae ────┘
```

---

## Примеры workflow

### Клонирование голоса

```text
prompt:  Диктор читает новости, профессиональный тон
ref_audio:  5-секундная запись целевого голоса
ref_image:  (опционально, но рекомендуется) портрет диктора или фон студии
```

### Звуковой дизайн с учётом окружения

```text
prompt:  Сильный дождь по жестяной крыше, далёкий раскат грома
ref_image:  Фото интерьера старой дачи
ref_audio:  (опционально) короткая запись дождя для соответствия текстуре
```

---

## Технические замечания

### Сетка кадров

Длина видео MiniMax H3 подчиняется правилу `17k + 5`. Запрошенный `length` округляется вверх через `align_frame_count`.

Примеры:

| Запрошено | Выровненный `frame_count` | ~Длительность @ 24 fps |
|-----------|---------------------------|------------------------|
| 124 | 124 | ~5.2 с |
| 200 | 209 | ~8.7 с |
| 362 | 362 | ~15.1 с |

Частота audio latent — **40 Hz** (800 сэмплов на latent-кадр при 32 kHz).

### Патч PackedLayout

При импорте патчится `PackedLayout.__init__`: временно подставляются нулевые индексы для прохождения проверки ядра, затем восстанавливаются реальные индексы и пересчитываются temporal `position_ids`.

### Формат аудио

Всегда:

```python
{
  "waveform": FloatTensor[B, C, L],  # обычно B=1, C=2 (стерео)
  "sample_rate": 32000
}
```

Совместимо с нодами **Preview Audio** / **Save Audio**.

### Размер референсного изображения

| Режим | Поведение | Когда использовать |
|-------|-----------|------------------|
| `match` | Масштабируется относительно canvas генерации (внутри 320×320) | Быстрые превью, приблизительное стилевое соответствие |
| `max` | Масштабируется до 2048 px по короткой стороне, затем crop до кратного canvas | Лучшее сохранение identity, больше VRAM |

---

## Решение проблем

| Проблема | Причина | Решение |
|----------|---------|---------|
| `sample() got an unexpected keyword argument 'device'` | Старая версия вызывала `comfy.sample.sample` с лишними kwargs | Обновить AIO-ноду (`sample_custom`) |
| `too many indices for tensor of dimension 3` на Save/Preview Audio | Вернули сырой тензор вместо AUDIO dict | Обновить AIO (`_to_audio_dict`) |
| Seed не меняется при randomize | Старый экземпляр ноды после смены INPUT_TYPES | Удалить ноду с графа, перезапустить ComfyUI, добавить заново |
| Offset игнорируется / ошибка ядра | Патч не применился | В консоли должно быть: `MiniMax H3 Keyframe Offset: Patch applied successfully` |
| Offset больше длины клипа | Ожидаемо | Значение clamp’ится; смотрите лог в консоли |
| Пустое / тихое аудио | Не тот VAE на `audio_vae` или denoise `0` | Подключите **Audio** VAE MiniMax; denoise `1.0` |
| Слабое клонирование / искажённый тембр | `ref_audio` без `ref_image` | Подключите `ref_image` (хотя бы нейтральный портрет); нативный H3 предпочитает визуальный якорь |
| Референсное изображение игнорируется | `use_references` выключен | Включите тумблер; входы появляются только при активации |
| Ошибки JSON у ComfyUI-Manager | Сеть / кэш Manager | К этому паку не относится |

---

## Лицензия

**Apache License, Version 2.0**. Подробности в [LICENSE](LICENSE).

```
Copyright 2026 asirusasr-maker / ComfyUI-MiniMax-H3-Keyframe-Offset contributors
```

---

## Благодарности

- Архитектура MiniMax H3 и официальная интеграция в ComfyUI: [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI), MiniMax
- Автор пака: [asirusasr-maker](https://github.com/asirusasr-maker)

---

*Репозиторий: [ComfyUI-MiniMax-H3-Keyframe-Offset](https://github.com/asirusasr-maker/ComfyUI-MiniMax-H3-Keyframe-Offset)*
