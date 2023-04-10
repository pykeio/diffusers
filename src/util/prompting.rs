//! Utilities for cleaning and modifying prompts.

use regex::{Captures, Regex};

/// Cleans up a potetnailly dirty prompt. This removes extraneous parentheses and commas, and cleans up trailing commas
/// and whitespace.
///
/// ```
/// # use pyke_diffusers::prompting::cleanup_prompt;
/// assert_eq!(
/// 	cleanup_prompt("(masterpiece,, best quality,:1.1)), 1girl,").as_str(),
/// 	"(masterpiece, best quality:1.1), 1girl"
/// );
/// ```
pub fn cleanup_prompt<S: AsRef<str>>(prompt: S) -> String {
	let split_regex: Regex = Regex::new(r#"\(*?(?:\([^)(]*(?:\([^)(]*(?:\([^)(]*(?:\([^)(]*\)[^)(]*)*\)[^)(]*)*\)[^)(]*)*\))\)*?|\b[^,]+\b"#).unwrap();
	let cleanup_emphasis_regex: Regex = Regex::new(r#"\(*?(\([^)(]*(?:\([^)(]*(?:\([^)(]*(?:\([^)(]*\)[^)(]*)*\)[^)(]*)*\)[^)(]*)*\))\)*"#).unwrap();
	let emphasis_trailing_comma_regex: Regex = Regex::new(r#"(\(+)([^:]*?),+(:\d[^)]+)?(\)+)"#).unwrap();
	let comma_regex: Regex = Regex::new(r#"\s*,+\s*"#).unwrap();
	let whitespace_regex: Regex = Regex::new(r#"\s+"#).unwrap();
	let trailing_leading_comma: Regex = Regex::new(r#"^,+\s*|,+\s*$"#).unwrap();

	fn emphasis_trailing_comma(cap: &Captures<'_>) -> String {
		cap.get(1).unwrap().as_str().to_owned() + cap.get(2).unwrap().as_str() + cap.get(3).unwrap().as_str() + cap.get(4).unwrap().as_str()
	}
	fn cleanup_emphasis(cap: &Captures<'_>) -> String {
		cap.get(1).unwrap().as_str().to_string()
	}
	fn cleanup_concept(cap: &Captures<'_>) -> String {
		cap.get(0).unwrap().as_str().trim().to_string()
	}

	let prompt = cleanup_emphasis_regex.replace_all(prompt.as_ref(), cleanup_emphasis);
	let prompt = emphasis_trailing_comma_regex.replace_all(prompt.as_ref(), emphasis_trailing_comma);
	let prompt = split_regex.replace_all(prompt.as_ref(), cleanup_concept);
	let prompt = comma_regex.replace_all(prompt.as_ref(), ", ");
	let prompt = whitespace_regex.replace_all(prompt.as_ref(), " ");
	let prompt = trailing_leading_comma.replace_all(prompt.as_ref(), "");
	prompt.trim().to_string()
}

/// Combines 2 concepts into one prompt.
///
/// The output prompt is only minimally cleaned (removing extraneous/trailing commas). You should pass the output prompt
/// into [`cleanup_prompt`] for best results.
///
/// ```
/// # use pyke_diffusers::prompting::combine_concepts;
/// assert_eq!(
/// 	combine_concepts("masterpiece, best quality,,", "1girl, solo, blue hair, ").as_str(),
/// 	"masterpiece, best quality, 1girl, solo, blue hair"
/// );
/// ```
pub fn combine_concepts<A: AsRef<str>, B: AsRef<str>>(a: A, b: B) -> String {
	let trailing_leading_comma: Regex = Regex::new(r#"^,+\s*|,+\s*$"#).unwrap();

	let a = trailing_leading_comma.replace_all(a.as_ref(), "");
	let b = trailing_leading_comma.replace_all(b.as_ref(), "");
	a.trim().to_string() + ", " + b.trim()
}

#[cfg(test)]
mod tests {
	use super::{cleanup_prompt, combine_concepts};

	#[test]
	fn test_cleanup_prompt() {
		assert_eq!(
			cleanup_prompt("(best quality,, masterpiece,:1.3)),  1girl, solo, blue hair, ").as_str(),
			"(best quality, masterpiece:1.3), 1girl, solo, blue hair"
		);
	}

	#[test]
	fn test_combine_concepts() {
		assert_eq!(combine_concepts("masterpiece, best quality,,", "1girl, solo, blue hair, ").as_str(), "masterpiece, best quality, 1girl, solo, blue hair");
	}
}
