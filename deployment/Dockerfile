# Use official PyTorch base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set workdir
WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .

# Expose port
EXPOSE 8501

# Start streamlit app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]