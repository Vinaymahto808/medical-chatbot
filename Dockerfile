FROM python:3.10-slim

# Install system dependencies (Git is required for some LangChain/GitHub installs)
RUN apt-get update && apt-get install -y git gcc python3-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Use Gunicorn for production instead of Flask's built-in server
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
