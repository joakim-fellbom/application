#/bin/bash

python3 train.py
uvicorn api:app --host "0.0.0.0"

chmod +x api/run.sh
