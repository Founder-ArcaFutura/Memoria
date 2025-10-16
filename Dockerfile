# Use official Python image as base
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy only the files needed first (to leverage Docker layer caching)
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port (matches your fly.toml)
EXPOSE 8000

# Command to run the app using a WSGI server
CMD ["gunicorn", "-b", "0.0.0.0:8000", "memoria_api:app"]
