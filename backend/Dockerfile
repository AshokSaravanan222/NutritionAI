# Stage 2: Setup the application environment
FROM python:3.6.1

# Update pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel


# Set environment variables for configuration
ENV INPUT_IMAGES="/path/to/image1 /path/to/image2" \
    DEPTH_MODEL_ARCHITECTURE="/app/models/depth_architecture.json" \
    DEPTH_MODEL_WEIGHTS="/app/models/depth_weights.h5" \
    SEGMENTATION_WEIGHTS="/app/models/segmentation_weights.h5" \
    FOV=70 \
    PLATE_DIAMETER_PRIOR=0.3 \
    GT_DEPTH_SCALE=0.35 \
    MIN_DEPTH=0.01 \
    MAX_DEPTH=10 \
    RELAXATION_PARAM=0.01 \
    PLOT_RESULTS=true \
    RESULTS_FILE="/app/results/results.csv" \
    PLOTS_DIRECTORY="/app/plots/" \
    DENSITY_DB="1yA4HU_yF8A6RwlY8x3eHi67YfdJKl3QYKhelQ60cnSQ" \
    FOOD_TYPE="exampleFoodType"

# Assuming tensorflow has been removed from requirements.txt manually
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install food-volume-estimation package
ADD food_volume_estimation ./food_volume_estimation/
COPY setup.py .
RUN python setup.py install

# Add model files and scripts to image
COPY models/fine_tune_food_videos/monovideo_fine_tune_food_videos.json $DEPTH_MODEL_ARCHITECTURE
COPY models/fine_tune_food_videos/monovideo_fine_tune_food_videos.h5 $DEPTH_MODEL_WEIGHTS
COPY models/segmentation/mask_rcnn_food_segmentation.h5 $SEGMENTATION_WEIGHTS
COPY food_volume_estimation/volume_estimator.py .
COPY test_imports.py .

# Run the test script as part of the Docker build process to ensure everything works
RUN python test_imports.py

# CMD to run the volume_estimator.py script with specified arguments
CMD python volume_estimator.py \
    --input_images $INPUT_IMAGES \
    --depth_model_architecture $DEPTH_MODEL_ARCHITECTURE \
    --depth_model_weights $DEPTH_MODEL_WEIGHTS \
    --segmentation_weights $SEGMENTATION_WEIGHTS \
    --fov $FOV \
    --plate_diameter_prior $PLATE_DIAMETER_PRIOR \
    --gt_depth_scale $GT_DEPTH_SCALE \
    --min_depth $MIN_DEPTH \
    --max_depth $MAX_DEPTH \
    --relaxation_param $RELAXATION_PARAM \
    --plot_results \
    --results_file $RESULTS_FILE \
    --plots_directory $PLOTS_DIRECTORY \
    --density_db $DENSITY_DB \
    --food_type $FOOD_TYPE
