# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Functions for logging."""

import logging
import sys


def get_logger(log_file):
  """Gets a logger given the path to the log file.

  Args:
    name: Path to the log file.

  Returns:
    A logger.
  """
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  formatter = logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S')

  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging.INFO)
  stdout_handler.setFormatter(formatter)
  logger.addHandler(stdout_handler)

  file_handler = logging.FileHandler(log_file, mode='w')
  file_handler.setLevel(logging.INFO)
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)

  return logger
