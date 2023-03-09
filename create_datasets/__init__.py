import sys
import logging
from logging import StreamHandler

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)

#Create the Handler for logging data to console.
console_handler = StreamHandler()
console_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)