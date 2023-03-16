FROM python:3.8.12-buster

COPY FastAPI.py FastAPI.py
COPY requirements.txt requirements.txt
COPY final_model.h5 final_model.h5

RUN pip install --upgrade pip wheel \
pip install -r requirements.txt

CMD uvicorn FastAPI:app --host 0.0.0.0 --port $PORT
