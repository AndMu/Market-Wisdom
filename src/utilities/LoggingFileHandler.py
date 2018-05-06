import logging
import os

from pathlib2 import Path
from os import makedirs, path


class LoggingFileHandler(logging.FileHandler):
    def __init__(self, dir_name, fileName, mode):

        if not Path(dir_name).exists():
            makedirs(dir_name)

        file_path = path.join(dir_name, str(os.getpid()) + "_" + fileName)
        super(LoggingFileHandler, self).__init__(file_path, mode)