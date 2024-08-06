# Use the official Python image from the Docker Hub
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

CMD ["python", "app.py"]

EXPOSE 8000

# Command to run the application using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:80", "app:server"]
