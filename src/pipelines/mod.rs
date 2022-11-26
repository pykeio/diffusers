//! Diffusion pipelines.

use std::{borrow::Cow, ops::Deref};

cfg_if::cfg_if! {
	if #[cfg(feature = "stable-diffusion")] {
		mod stable_diffusion;
		pub use self::stable_diffusion::*;
	}
}

/// Text prompt(s) used as input in diffusion pipelines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt(pub(crate) Vec<String>);

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

macro_rules! from_slice {
	($($size: literal),*) => {
		$(
			impl<'s> From<[&'s str; $size]> for Prompt {
				fn from(value: [&'s str; $size]) -> Self {
					Self(value.iter().map(|v| v.to_string()).collect())
				}
			}
		)*
	};
}

from_slice!(1, 2, 3, 4, 5, 6, 7, 8);

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
