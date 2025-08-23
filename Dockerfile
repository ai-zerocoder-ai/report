FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
RUN mkdir -p /app/chroma_db

EXPOSE 5000

CMD ["python", "src/app.py"]




#FROM python:3.11-slim

#WORKDIR /app
#ENV PYTHONUNBUFFERED=1

#RUN pip install --upgrade pip

#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt && pip install gunicorn

#COPY src/ ./src/
#COPY data/ ./data/
#RUN mkdir -p /app/chroma_db

#EXPOSE 5000
# 1 воркер + 4 треда — достаточно для маленького сервиса
#CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:5000", "src.app:create_app()"]
