import argparse
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Model, model_from_json
from food_volume_estimation.volume_estimator import VolumeEstimator, DensityDatabase
from food_volume_estimation.depth_estimation.custom_modules import *
from food_volume_estimation.food_segmentation.food_segmentator import FoodSegmentator
from flask import Flask, request, jsonify, make_response, abort
import base64

import logging
import sys

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensures logs are printed to stdout
    ]
)

app = Flask(__name__)
estimator = None
density_db = None


def load_volume_estimator(depth_model_architecture, depth_model_weights,
        segmentation_model_weights, density_db_source):
    """Loads volume estimator object and sets up its parameters."""
    logging.info('[*] Loading volume estimator...' + str(depth_model_architecture) + str(depth_model_weights) + str(segmentation_model_weights) + str(density_db_source))
    # Create estimator object and intialize
    global estimator
    estimator = VolumeEstimator(arg_init=False)
    with open(depth_model_architecture, 'r') as read_file:
        custom_losses = Losses()
        objs = {'ProjectionLayer': ProjectionLayer,
                'ReflectionPadding2D': ReflectionPadding2D,
                'InverseDepthNormalization': InverseDepthNormalization,
                'AugmentationLayer': AugmentationLayer,
                'compute_source_loss': custom_losses.compute_source_loss}
        model_architecture_json = json.load(read_file)
        estimator.monovideo = model_from_json(model_architecture_json,
                                              custom_objects=objs)
        logging.info('[*] Loaded depth estimation model architecture.')
    estimator._VolumeEstimator__set_weights_trainable(estimator.monovideo,
                                                      False)
    estimator.monovideo.load_weights(depth_model_weights)
    estimator.model_input_shape = (
        estimator.monovideo.inputs[0].shape.as_list()[1:])
    depth_net = estimator.monovideo.get_layer('depth_net')
    logging.info('[*] Loaded depth estimation model weights.')
    estimator.depth_model = Model(inputs=depth_net.inputs,
                                  outputs=depth_net.outputs,
                                  name='depth_model')
    logging.info('[*] Loaded depth estimation model.')

    # Depth model configuration
    MIN_DEPTH = 0.01
    MAX_DEPTH = 10
    estimator.min_disp = 1 / MAX_DEPTH
    estimator.max_disp = 1 / MIN_DEPTH
    estimator.gt_depth_scale = 0.35 # Ground truth expected median depth

    # Create segmentator object
    estimator.segmentator = FoodSegmentator(segmentation_model_weights)
    # Set plate adjustment relaxation parameter
    estimator.relax_param = 0.01

    # Need to define default graph due to Flask multiprocessing
    global graph
    graph = tf.get_default_graph()

    # Load food density database
    global density_db
    density_db = DensityDatabase(density_db_source)
    logging.info('[*] Loaded food density database.')

@app.route('/test')
def test_endpoint():
    """A simple test endpoint."""
    return "Hello World"

@app.route('/predict', methods=['POST'])
def volume_estimation():
    """Receives an HTTP POST request with JSON body and returns the estimated 
    volumes of the foods in the image given.

    JSON body format:
    {
        "img": "base64_encoded_image",
        "plate_diameter": "expected_plate_diameter",
        "food_type": "food_name"
    }

    Returns:
        The array of estimated volumes in JSON format.
    """
    
    # Decode incoming JSON to get an image and other data
    try:
        content = request.get_json()
        img_encoded = content['img']
        img_data = base64.b64decode(img_encoded)
        np_img = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except Exception as e:
        abort(406, description="Invalid image data")

    # Extract food type
    try:
        food_type = content['food_type']
    except KeyError:
        abort(406, description="Food type not provided")

    # Extract expected plate diameter from JSON data or set to 0 and ignore
    plate_diameter = content.get('plate_diameter', 0)
    try:
        plate_diameter = float(plate_diameter)
    except ValueError:
        abort(406, description="Invalid plate diameter")

    # Estimate volumes
    with graph.as_default():
        volumes = estimator.estimate_volume(img, fov=70,
                                            plate_diameter_prior=plate_diameter)
    # Convert to mL
    volumes = [v * 1e6 for v in volumes]
    
    # Convert volumes to weight - assuming a single food type
    db_entry = density_db.query(food_type)
    if db_entry is None or len(db_entry) == 0:
        abort(404, description="Food type not found in database")
    density = db_entry[1]
    weight = sum(v * density for v in volumes)

    # Return values
    return_vals = {
        'food_type_match': db_entry[0],
        'weight': weight
    }
    return make_response(jsonify(return_vals), 200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Food volume estimation API.')
    parser.add_argument('--depth_model_architecture', type=str,
                        help='Path to depth model architecture (.json).',
                        metavar='/path/to/architecture.json',
                        required=True)
    parser.add_argument('--depth_model_weights', type=str,
                        help='Path to depth model weights (.h5).',
                        metavar='/path/to/depth/weights.h5',
                        required=True)
    parser.add_argument('--segmentation_model_weights', type=str,
                        help='Path to segmentation model weights (.h5).',
                        metavar='/path/to/segmentation/weights.h5',
                        required=True)
    parser.add_argument('--density_db_source', type=str,
                        help=('Path to food density database (.xlsx) ' +
                              'or Google Sheets ID.'),
                        metavar='/path/to/plot/database.xlsx or <ID>',
                        required=True)
    args = parser.parse_args()
    
    estimator = VolumeEstimator(arg_init=False)
    with open(args.depth_model_architecture, 'r') as read_file:
        custom_losses = Losses()
        objs = {'ProjectionLayer': ProjectionLayer,
                'ReflectionPadding2D': ReflectionPadding2D,
                'InverseDepthNormalization': InverseDepthNormalization,
                'AugmentationLayer': AugmentationLayer,
                'compute_source_loss': custom_losses.compute_source_loss}
        model_architecture_json = json.load(read_file)
        estimator.monovideo = model_from_json(model_architecture_json,
                                              custom_objects=objs)
        logging.info('[*] Loaded depth estimation model architecture.')
    
    estimator.segmentator = FoodSegmentator(args.segmentation_model_weights)
    logging.info('[*] Loaded segmentation model weights.')

    load_volume_estimator(args.depth_model_architecture,
                          args.depth_model_weights, 
                          args.segmentation_model_weights,
                          args.density_db_source)
    app.run(host='0.0.0.0')

