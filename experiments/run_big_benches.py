import argparse
import numpy as np
from dvclive import Live
import faiss
import os
import logging
import platform, socket, psutil


ENVIRONMENT_INFO = {
    "load_before_experiment": os.getloadavg()[2],
    "platform": platform.platform(),
    "platform_version": platform.version(),
    "platform_release": platform.release(),
    "architecture": platform.machine(),
    "processor": platform.processor(),
    "hostname": socket.gethostname(),
    "ram_gb": round(psutil.virtual_memory().total / (1024.0**3)),
    "num_cores": psutil.cpu_count(logical=True),
}
