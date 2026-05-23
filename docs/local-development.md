# Локальная разработка

Эта инструкция описывает, как запускать проект локально на macOS.

## Корень проекта

Сначала перейди в корень проекта:

```bash
cd /Users/suusumi/Projects/segmentation-classification-neutrophils
```

## Python-окружение

Активируй виртуальное окружение:

```bash
source .venv/bin/activate
```

Проверь, что используется Python 3.11:

```bash
python --version
```

Ожидаемый результат:

```text
Python 3.11.x
```

Если папки `.venv` нет, создай окружение заново:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Backend

FastAPI backend запускается из корня проекта:

```bash
uvicorn src.api.app:app --reload
```

API будет доступен по адресу:

```text
http://127.0.0.1:8000
```

Проверка, что backend работает:

```bash
curl http://127.0.0.1:8000/health
```

Ожидаемый ответ:

```json
{"status":"ok","service":"Neutrophil Analysis","version":"0.1.0"}
```

## Frontend

Открой второй терминал и запусти:

```bash
cd /Users/suusumi/Projects/segmentation-classification-neutrophils/frontend
nvm use
npm run dev
```

Приложение будет доступно по адресу:

```text
http://127.0.0.1:5173
```

Если frontend-зависимости не установлены:

```bash
nvm use
npm install
```

## Тесты и проверки

Команды ниже запускаются из корня проекта с активированным `.venv`.

Запуск тестов:

```bash
pytest
```

Проверка стиля кода:

```bash
ruff check .
```

Проверка типов:

```bash
mypy src tests scripts
```

Проверка production-сборки frontend:

```bash
cd frontend
nvm use
npm run build
```

## Docker

Запуск backend и frontend через Docker:

```bash
docker compose up --build
```

Сервисы будут доступны по адресам:

```text
Backend:  http://127.0.0.1:8000
Frontend: http://127.0.0.1:5173
```

Остановить Docker-сервисы:

```bash
docker compose down
```

## Остановка локальных dev-серверов

Если backend или frontend запущены напрямую через `uvicorn` или `npm run dev`, остановить сервер можно сочетанием:

```text
Ctrl+C
```

