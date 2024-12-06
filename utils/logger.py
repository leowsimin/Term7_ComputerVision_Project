import logging, os
from datetime import datetime
import sys
import tensorflow as tf
import config

# create folder 'logs' if not exist
folder_name = "logs"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created.")
else:
    print(f"Folder '{folder_name}' already exists.")

# get TF logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
filename = f"{datetime.today().strftime('%Y_%m_%d_%H_%M_%S')}"
fh = logging.FileHandler(f'logs/{filename}.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# log config values
logger.info("Config:")
for k, v in vars(config).items():
    if k[:2] != '__':
        logger.info(f"{k}={v}")

# save model training + predictions logs to file
class LogCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        logger.info("Starting training; got log: {}".format(logs))

    def on_train_end(self, logs=None):
        logger.info("Stop training; got log: {}".format(logs))

    def on_epoch_begin(self, epoch, logs=None):
        logger.info("Start epoch {} of training; got log: {}".format(epoch, logs))

    def on_epoch_end(self, epoch, logs=None):
        logger.info("End epoch {} of training; got log: {}".format(epoch, logs))

    def on_predict_begin(self, logs=None):
        logger.info("Start predicting; got log: {}".format(logs))

    def on_predict_end(self, logs=None):
        logger.info("Stop predicting; got log: {}".format(logs))    
        
    def on_test_begin(self, logs=None):
        logger.info("Start testing; got log: {}".format(logs))

    def on_test_end(self, logs=None):
        logger.info("Stop testing; got log: {}".format(logs))

keras_custom_callback = LogCallback()

# log model archi
def _replace_special_unicode_character(message):
    message = str(message).replace("━", "=")  # Fall back to Keras2 behavior.
    return message

def print_and_log(message, line_break=True):
    message = str(message)
    message = message + "\n" if line_break else message
    try:
        # sys.stdout.write(message)
        logger.info(message)
    except UnicodeEncodeError:
        # If the encoding differs from UTF-8, `sys.stdout.write` may fail.
        # To address this, replace special unicode characters in the
        # message, and then encode and decode using the target encoding.
        message = _replace_special_unicode_character(message)
        message_bytes = message.encode(sys.stdout.encoding, errors="ignore")
        message = message_bytes.decode(sys.stdout.encoding)
    sys.stdout.write(message)