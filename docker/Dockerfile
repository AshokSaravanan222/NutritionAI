# Use an official Python runtime as a parent image
FROM python:3.6.15-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install NumPy version 1.16.3
RUN pip install numpy==1.16.3

# This command runs when the container starts. Here, it's used to copy the installed NumPy package to the mounted volume.
CMD ["cp", "-r", "/usr/local/lib/python3.6/site-packages/numpy", "/data/numpy"]
