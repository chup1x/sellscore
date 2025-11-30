FROM python:3.11-slim

# Устанавливаем системные библиотеки для LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Создаём директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Ставим python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

EXPOSE 8000

# Запуск API
CMD ["python", "simple_api.py"]
