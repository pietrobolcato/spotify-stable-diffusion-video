#!/bin/bash
gunicorn -w "${WORKERS:=2}" -b :80 -t "${TIMEOUT:=300}" -k uvicorn.workers.UvicornWorker inference_server.app:app
