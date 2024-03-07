# test_imports.py
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensures logs are printed to stdout
    ]
)

logging.info("Printing all imports")

import os
logging.info("imported os")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import argparse
logging.info("imported argparse")
import numpy as np
logging.info("imported numpy")
import pandas as pd
logging.info("imported pandas")
import cv2
logging.info("imported cv2")
import json
logging.info("imported json")
from scipy.spatial.distance import pdist
logging.info("imported pdist")
from scipy.stats import skew
logging.info("imported skew")
from fuzzywuzzy import fuzz, process 
logging.info("imported fuzzywuzzy")
import matplotlib.pyplot as plt
logging.info("imported matplotlib")

try:
    from keras.models import Model, model_from_json
    logging.info("imported keras")
except Exception as e:
    logging.error(f"Failed to import Keras: {e}")

try:
    import tensorflow as tf
    logging.info("imported tensorflow")
except Exception as e:
    logging.error(f"Failed to import TensorFlow: {e}")

# Print versions (if applicable) to verify installations
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"Pandas version: {pd.__version__}")
# Note: Some packages like fuzzywuzzy might not have a __version__ attribute

# Add any additional tests you deem necessary for your application
