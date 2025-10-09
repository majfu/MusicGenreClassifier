#!/bin/bash

cd /app
python3 -m backend.app &

nginx -g 'daemon off;'