FROM python:3.9

COPY requirements/prod.txt .

RUN pip install --upgrade pip && pip install -r prod.txt

WORKDIR /app
COPY . /app
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "./app.app:app", "--host", "0.0.0.0"]