web: gunicorn --bind 0.0.0.0:$PORT stock_analysis:app --timeout 900 --workers 1 --threads 1 --max-requests 50 --max-requests-jitter 10 --worker-class sync --preload --worker-tmp-dir /dev/shm
