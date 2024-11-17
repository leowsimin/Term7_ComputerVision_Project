import logging, os
from datetime import datetime
import config
import tensorflow as tf

# create folder 'logs' if not
folder_name = "logs"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created.")
else:
    print(f"Folder '{folder_name}' already exists.")

# get TF logger
tf_logger = tf.get_logger()
tf_logger.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
filename = f'{datetime.today().strftime('%Y_%m_%d_%H_%M_%S')}.log'
fh = logging.FileHandler(f'logs/{filename}')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
tf_logger.addHandler(fh)

# log config values
tf_logger.info("Config:")
for k, v in vars(config).items():
    if k[:2] != '__':
        tf_logger.info(f"{k}={v}")