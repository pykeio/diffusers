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

//! Diffusion pipelines.

use std::{borrow::Cow, ops::Deref};

cfg_if::cfg_if! {
	if #[cfg(feature = "stable-diffusion")] {
		mod stable_diffusion;
		pub use self::stable_diffusion::*;
	}
}

/// Text prompt(s) used as input in diffusion pipelines.
///
/// Can be converted from one or more prompts:
/// ```no_run
/// # use pyke_diffusers::Prompt;
/// let prompt: Prompt = "photo of a red fox".into();
/// let prompts: Prompt = ["photo of a red fox", "photo of an Arctic fox"].into();
/// let prompts: Prompt = vec!["photo of a red fox", "photo of an Arctic fox"].into();
/// ```
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Prompt(pub(crate) Vec<String>);

impl Prompt {
	/// Creates a default prompt with a given batch size.
	pub fn default_batched(batch_size: usize) -> Prompt {
		Prompt(vec![String::new(); batch_size])
	}

	/// Converts this prompt into a batched prompt with `batch_size` batches.
	pub fn batched(mut self, batch_size: usize) -> Prompt {
		assert!(self.0.len() == 1);
		self.0 = vec![self.0[0].clone(); batch_size];
		self
	}
}

impl Deref for Prompt {
	type Target = Vec<String>;

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl<'s> From<&'s [&'s str]> for Prompt {
	fn from(value: &'s [&'s str]) -> Self {
		Self(value.iter().map(|v| v.to_string()).collect())
	}
}

impl<'s, const N: usize> From<[&'s str; N]> for Prompt {
	fn from(value: [&'s str; N]) -> Self {
		Self(value.iter().map(|v| v.to_string()).collect())
	}
}

impl<'s> From<&'s [Cow<'s, str>]> for Prompt {
	fn from(value: &'s [Cow<'s, str>]) -> Self {
		Self(value.iter().map(|v| v.to_string()).collect())
	}
}

impl<'s> From<&'s [String]> for Prompt {
	fn from(value: &'s [String]) -> Self {
		Self(value.to_vec())
	}
}

impl<'s> From<&'s str> for Prompt {
	fn from(value: &'s str) -> Self {
		Self(vec![value.to_string()])
	}
}

impl<'s> From<Cow<'s, str>> for Prompt {
	fn from(value: Cow<'s, str>) -> Self {
		Self(vec![value.to_string()])
	}
}

impl From<String> for Prompt {
	fn from(value: String) -> Self {
		Self(vec![value])
	}
}

impl<'s> From<&'s String> for Prompt {
	fn from(value: &'s String) -> Self {
		Self(vec![value.to_string()])
	}
}

impl<'s> From<Vec<&'s str>> for Prompt {
	fn from(value: Vec<&'s str>) -> Self {
		Self(value.iter().map(|v| v.to_string()).collect())
	}
}

impl<'s> From<Vec<Cow<'s, str>>> for Prompt {
	fn from(value: Vec<Cow<'s, str>>) -> Self {
		Self(value.iter().map(|v| v.to_string()).collect())
	}
}

impl From<Vec<String>> for Prompt {
	fn from(value: Vec<String>) -> Self {
		Self(value)
	}
}
