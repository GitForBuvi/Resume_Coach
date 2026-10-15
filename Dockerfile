# 1. Start with clean, official Python 3.11 Linux image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy dependencies first
COPY requirements.txt .

# 4. Install all packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of our application files
COPY . .

# 6. Expose the port Streamlit uses
EXPOSE 8501

# 7. Start the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]