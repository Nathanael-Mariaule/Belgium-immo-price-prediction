FROM python:3.8
RUN apt-get update
RUN mkdir /app
COPY /content /app
RUN python3 -m venv /opt/venv
COPY requirements.txt .
RUN . /opt/venv/bin/activate && pip install -r requirements.txt
WORKDIR /app
EXPOSE 8501
CMD . /opt/venv/bin/activate && exec streamlit run app.py

