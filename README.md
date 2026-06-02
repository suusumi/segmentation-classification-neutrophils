# Сегментация и классификация нейтрофилов

Проект анализирует изображения одиночных нейтрофилов: выделяет ядро, оценивает сегменты ядра, извлекает морфологические признаки и формирует вывод о нормальной сегментации или гиперсегментации. Результаты сохраняются как артефакты анализа: маски, overlay-изображения, JSON/Markdown-отчет, лог и запись в SQLite.

В репозитории есть:

- FastAPI backend для загрузки изображения и запуска анализа.
- React/Vite frontend для работы через браузер.
- ML-пайплайн для предобработки, сегментации, постобработки, подсчета сегментов и классификации.
- Скрипты для подготовки датасетов, разметки, обучения и оценки моделей.
- Docker Compose для запуска backend и frontend одной командой.

## Структура проекта

```text
.
├── src/
│   ├── api/          FastAPI, схемы ответов и HTTP endpoints
│   ├── core/         настройки и пути проекта
│   ├── datasets/     загрузчики датасетов и трансформации
│   ├── detection/    YOLO-детектор долей ядра
│   ├── models/       архитектуры моделей
│   ├── pipeline/     основной пайплайн анализа изображения
│   ├── services/     orchestration, валидация и доменные ошибки
│   ├── storage/      SQLite-хранилище результатов анализа
│   ├── training/     метрики и postprocessing для обучения
│   └── utils/        конфигурация, IO, логирование и seed helpers
├── scripts/          CLI-скрипты для датасетов, разметки, обучения и оценки
├── frontend/         React/Vite интерфейс
├── configs/          конфигурационные файлы
├── docs/             инструкции по workflow проекта
├── data/             входные, промежуточные и обработанные данные
├── models/           веса моделей для локального и Docker-запуска
├── outputs/          результаты анализов, отчеты и логи
├── Dockerfile        Docker image для backend
└── docker-compose.yml
```

Основные API endpoints:

```text
GET  /health
POST /analysis
POST /analysis/yolo
GET  /analysis/{analysis_id}
GET  /analysis/{analysis_id}/logs
GET  /analysis/{analysis_id}/report
GET  /artifacts/{artifact_path}
```

## Требования

- Python 3.11.
- Node.js 22 для frontend.
- Docker и Docker Compose для запуска в контейнерах.

Для локального backend-запуска нужны Python-зависимости из `requirements.txt`. Для Docker backend использует `requirements.runtime.txt` и CPU-сборку PyTorch.

## Запуск локально на Windows

Команды ниже выполняются из корня репозитория в PowerShell.

### Backend

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Проверка:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Пример загрузки изображения:

```powershell
curl.exe -X POST http://localhost:8000/analysis `
  -F "file=@data/raw/acevedo/neutrophil/NEUTROPHIL_100292.jpg"
```

### Frontend

Откройте второй PowerShell:

```powershell
cd frontend
npm ci
$env:VITE_API_URL = "http://localhost:8000"
npm run dev -- --host 0.0.0.0
```

Интерфейс будет доступен по адресу http://localhost:5173.

## Запуск локально на macOS

Команды ниже выполняются из корня репозитория.

### Backend

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl http://localhost:8000/health
```

Пример загрузки изображения:

```bash
curl -X POST http://localhost:8000/analysis \
  -F "file=@data/raw/acevedo/neutrophil/NEUTROPHIL_100292.jpg"
```

### Frontend

Откройте второй терминал:

```bash
cd frontend
npm ci
VITE_API_URL=http://localhost:8000 npm run dev -- --host 0.0.0.0
```

Интерфейс будет доступен по адресу http://localhost:5173.

## Запуск локально на Linux

Команды ниже выполняются из корня репозитория. Для OpenCV в некоторых дистрибутивах нужны системные библиотеки:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

### Backend

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl http://localhost:8000/health
```

Пример загрузки изображения:

```bash
curl -X POST http://localhost:8000/analysis \
  -F "file=@data/raw/acevedo/neutrophil/NEUTROPHIL_100292.jpg"
```

### Frontend

Откройте второй терминал:

```bash
cd frontend
npm ci
VITE_API_URL=http://localhost:8000 npm run dev -- --host 0.0.0.0
```

Интерфейс будет доступен по адресу http://localhost:5173.

## Запуск через Docker

Из корня репозитория:

```bash
docker compose up --build
```

После запуска:

- backend: http://localhost:8000
- frontend: http://localhost:5173
- health check: http://localhost:8000/health

`docker-compose.yml` монтирует локальные директории в контейнер backend:

```text
./data    -> /app/data
./models  -> /app/models
./outputs -> /app/outputs
```

Это позволяет сохранять результаты анализа и использовать локальные веса моделей без пересборки образа.

Остановить контейнеры:

```bash
docker compose down
```

## Тесты

```bash
pytest
```

## Дополнительная документация

- `docs/annotation-workflow.md` - workflow разметки масок ядра.
- `docs/lobe-annotation-workflow.md` - workflow разметки долей ядра.
- `docs/yolo-lobe-workflow.md` - обучение YOLO для долей ядра и preannotations.
- `docs/local-development.md` - дополнительные заметки по локальной разработке.
