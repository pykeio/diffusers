//! CLIP tokenizer implementation.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tokenizers::{models::bpe::BPE, EncodeInput, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

#[derive(Serialize, Deserialize)]
pub struct CLIPStandardTokenizerWrapper {
	#[serde(flatten)]
	pub tokenizer: Tokenizer,
	pub model_max_length: usize,
	pub bos_token_id: u32,
	pub eos_token_id: u32
}

/// A basic [CLIP](https://arxiv.org/abs/2103.00020) tokenizer.
///
/// CLIP is used by many diffusion models, including Stable Diffusion, for prompt tokenization and feature extraction.
pub struct CLIPStandardTokenizer(CLIPStandardTokenizerWrapper);

unsafe impl Send for CLIPStandardTokenizer {}
unsafe impl Sync for CLIPStandardTokenizer {}

impl CLIPStandardTokenizer {
	/// Loads a CLIP tokenizer from a file.
	pub fn new(path: impl Into<PathBuf>) -> anyhow::Result<Self> {
		let path = path.into();
		let bytes = std::fs::read(path)?;
		Self::from_bytes(bytes)
	}

	/// Loads a CLIP tokenizer from a byte array.
	pub fn from_bytes<B: AsRef<[u8]>>(bytes: B) -> anyhow::Result<Self> {
		let mut wrapper: CLIPStandardTokenizerWrapper = serde_json::from_slice(bytes.as_ref())?;
		// For some reason, CLIP tokenizers lose their padding and truncation config when converting from the old HF tokenizers
		// format, so we have to add them back here.
		wrapper
			.tokenizer
			.with_padding(Some(PaddingParams {
				strategy: PaddingStrategy::Fixed(wrapper.model_max_length),
				// `clip-vit-base-patch32` and (maybe) all Stable Diffusion models use `"pad_token": "<|endoftext|>"`
				// This info is also lost in translation in HF tokenizers.
				pad_id: wrapper.eos_token_id,
				..Default::default()
			}))
			.with_truncation(Some(TruncationParams {
				max_length: wrapper.model_max_length,
				..Default::default()
			}));
		Ok(Self(wrapper))
	}

	/// Returns the BPE model used by the tokenizer.
	///
	/// # Panics
	///
	/// - Panics with `unreachable!()` if the tokenizer model is not BPE (which would mean this tokenizer isn't a CLIP
	///   tokenizer, which should be impossible).
	#[allow(dead_code)]
	pub fn model(&self) -> &BPE {
		match self.0.tokenizer.get_model() {
			tokenizers::ModelWrapper::BPE(ref bpe) => bpe,
			_ => unreachable!()
		}
	}

	/// Returns the maximum length of tokens this tokenizer supports. For most CLIP models, this is 77 tokens.
	#[allow(clippy::len_without_is_empty)]
	pub fn len(&self) -> usize {
		self.0.model_max_length
	}

	/// Returns the ID of the end-of-string token.
	#[allow(dead_code)]
	pub fn eos(&self) -> u32 {
		self.0.eos_token_id
	}

	/// Returns the ID of the beginning-of-string token.
	#[allow(dead_code)]
	pub fn bos(&self) -> u32 {
		self.0.bos_token_id
	}

	/// Encodes the input string(s) into an array of token IDs.
	pub fn encode<'s, 'e, E>(&self, enc: Vec<E>) -> anyhow::Result<Vec<Vec<u32>>>
	where
		E: Into<EncodeInput<'s>>
	{
		let enc_len = enc.len();
		let encoded: Vec<Vec<u32>> = enc
			.into_iter()
			.map(|f| self.0.tokenizer.encode(f, true).map(|f| f.get_ids().to_vec()))
			.scan((), |_, x| x.ok())
			.collect();
		assert_eq!(encoded.len(), enc_len);
		Ok(encoded)
	}
}
