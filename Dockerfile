FROM python: latest

WORKDIR /myapp

COPY requitrements.txt .

RUN pip install -r requitrements.txt

COPY app.py .

CMD ["python", "app.py"]