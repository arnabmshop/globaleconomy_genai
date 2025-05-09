# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the local requirements.txt file to the container's working directory
COPY requirements.txt .

# Install dependencies inside the container
RUN pip install --no-cache-dir -r requirements.txt

# If using unzipped vectorstores, copy them directly into the container
COPY vectorstores /app/vectorstores
COPY imf_excel_vectorstore /app/imf_excel_vectorstore

# Copy the rest of the application code into the container
COPY app.py tools.py utils.py summarization_utils.py /app/

# Expose port 8080 (Cloud Run expected port)
EXPOSE 8080

# Start Streamlit and ensure it listens on port 8080 and all network interfaces
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
