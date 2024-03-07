#!/bin/bash

# Print environment variables (for debugging purposes)
echo "Starting volume_estimator.py with the following environment variables:"
printenv

# Check if the plots directory exists; create if not
if [ ! -d "$PLOTS_DIRECTORY" ]; then
  mkdir -p "$PLOTS_DIRECTORY"
fi

# Now execute the Python script with the arguments
exec python volume_estimator.py \
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
