ROM python
RUN apt-get update && apt-get install -y cmake

WORKDIR /app

# Install requirements as a separate step to encourage cacheing.
COPY requirements.txt /app
RUN pip install --requirement requirements.txt

CMD python -u main_pong.py
COPY ./* ./
