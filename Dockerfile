FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3 python3-pip nginx curl git ffmpeg

RUN curl -sL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

RUN npm install -g vite

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

WORKDIR /app/frontend
RUN npm install && npm run build

COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 5000 80

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]