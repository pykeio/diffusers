from datetime import date
import errno
import gc
import os
from pathlib import Path

import torch
from yaspin import spinners

SPINNER = spinners.Spinners._asdict()['clock' if date.today().month == 3 and date.today().day == 14 else 'dots12']
PROJECT_ROOT = Path(os.path.dirname(os.path.realpath(__file__))).parent

def collect_garbage():
	torch.cuda.empty_cache()
	gc.collect()

def mkdirp(path: Path):
	try:
		os.makedirs(path)
	except OSError as e:
		if e.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise e
