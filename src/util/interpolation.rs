// Copyright 2022-2023 pyke.io
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![allow(unused)]

use ndarray::ArrayView1;

pub struct InterpolationAccelerator {
	pub(crate) cache: usize,
	pub(crate) hit_count: usize,
	pub(crate) miss_count: usize
}

impl Default for InterpolationAccelerator {
	fn default() -> Self {
		Self::new()
	}
}

impl InterpolationAccelerator {
	pub fn new() -> InterpolationAccelerator {
		InterpolationAccelerator {
			cache: 0,
			hit_count: 0,
			miss_count: 0
		}
	}
}

#[allow(dead_code)]
pub(super) fn bsearch(xarr: &ArrayView1<'_, f32>, x: f32, idx_low: usize, idx_high: usize) -> usize {
	let mut ilow = idx_low;
	let mut ihigh = idx_high;

	while ihigh > ilow + 1 {
		let i = (ihigh + ilow) / 2;
		if xarr[i] > x {
			ihigh = i;
		} else {
			ilow = i;
		}
	}
	ilow
}

#[allow(dead_code)]
pub(super) fn accel_find(xarr: &ArrayView1<'_, f32>, x: f32, acc: &mut InterpolationAccelerator) -> usize {
	let xidx = acc.cache;

	if x < xarr[xidx] {
		acc.miss_count += 1;
		acc.cache = bsearch(xarr, x, 0, xidx);
	} else if x >= xarr[xidx + 1] {
		acc.miss_count += 1;
		acc.cache = bsearch(xarr, x, xidx, xarr.len() - 1);
	} else {
		acc.hit_count += 1;
	}
	acc.cache
}

pub struct LinearInterpolatorAccelerated<'x, 'y> {
	x: ArrayView1<'x, f32>,
	y: ArrayView1<'y, f32>,
	acc: InterpolationAccelerator
}

impl<'x, 'y> LinearInterpolatorAccelerated<'x, 'y> {
	pub fn new(x: ArrayView1<'x, f32>, y: ArrayView1<'y, f32>) -> Self {
		Self {
			x,
			y,
			acc: InterpolationAccelerator::default()
		}
	}

	fn accel_find(&mut self, x: f32) -> usize {
		accel_find(&self.x, x, &mut self.acc)
	}

	pub fn eval(&mut self, x: f32) -> f32 {
		if x < self.x[0] || x > self.x[self.x.len() - 1] {
			return f32::NAN;
		}

		let idx = self.accel_find(x);

		let x_l = self.x[idx];
		let x_h = self.x[idx + 1];
		let y_l = self.y[idx];
		let y_h = self.y[idx + 1];
		let dx = x_h - x_l;
		if dx > 0.0 { y_l + (x - x_l) / dx * (y_h - y_l) } else { f32::NAN }
	}
}
