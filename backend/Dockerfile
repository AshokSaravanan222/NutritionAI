# Stage 2: Setup the application environment
FROM python:3.6.1

# Update pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel
# Assuming tensorflow has been removed from requirements.txt manually
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install food-volume-estimation package
ADD food_volume_estimation/ food_volume_estimation/
COPY setup.py .
RUN python setup.py install

# Add model files and scripts to image
COPY models/fine_tune_food_videos/monovideo_fine_tune_food_videos.json models/depth_architecture.json
COPY models/fine_tune_food_videos/monovideo_fine_tune_food_videos.h5 models/depth_weights.h5
COPY models/segmentation/mask_rcnn_food_segmentation.h5 models/segmentation_weights.h5
COPY test_imports.py .

# Run the test script as part of the Docker build process to ensure everything works
RUN python test_imports.py

# Expose port 5000 for the Flask application
EXPOSE 5000

# Copy and execute server script
COPY food_volume_estimation_app.py .
ENTRYPOINT ["python", "food_volume_estimation_app.py", \
            "--depth_model_architecture", "models/depth_architecture.json", \
            "--depth_model_weights", "models/depth_weights.h5", \
            "--segmentation_model_weights", "models/segmentation_weights.h5", \
            "--density_db_source", "1yA4HU_yF8A6RwlY8x3eHi67YfdJKl3QYKhelQ60cnSQ"]
