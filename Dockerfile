# Using an official Python runtime as a base image
FROM python:3.11-slim

# Setting the working directory 
WORKDIR /app

# Copying the current directory contents into the container at /app
COPY . /app

# Installing any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Exposing port 8080 for Flask
EXPOSE 8080

# Set environment variables for Flask
ENV FLASK_APP=client_side/api.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run Flask in production mode
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
