# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the local requirements.txt file to the container's working directory
COPY requirements.txt . 

# Install dependencies inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ZIP files into the container
COPY vectorstores.zip /app/
COPY imf_excel_vectorstore.zip /app/

# Install unzip utility to extract the ZIP files
RUN apt-get update && apt-get install -y unzip

# Unzip the vectorstores.zip file
RUN unzip /app/vectorstores.zip -d /app/vectorstores

# Unzip the imf_excel_vectorstore.zip file
RUN unzip /app/imf_excel_vectorstore.zip -d /app/imf_excel_vectorstore

# Copy the rest of the application code into the container
COPY app.py tools.py utils.py summarization_utils.py /app/

# Expose port 8501 (default for Streamlit)
EXPOSE 8501

# Start Streamlit when the container runs
CMD ["streamlit", "run", "app.py"]