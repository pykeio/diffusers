# Copyright 2022-2023 pyke.io
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
