# startup.sh
# Install necessary system libraries for WeasyPrint
apt-get update && apt-get install -y libglib2.0-0 libglib2.0-dev libpango-1.0-0 libpangoft2-1.0-0 libjpeg-dev libopenjp2-7-dev libffi-dev


# Start the Gunicorn server
gunicorn --bind=0.0.0.0:8000 wsgi:server
