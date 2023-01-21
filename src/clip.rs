//! CLIP tokenizer implementation.

use std::path::PathBuf;

use ndarray::Array2;
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
pub struct CLIPStandardTokenizer {
	pub tokenizer: Tokenizer,
	model_max_length: usize,
	bos_token_id: u32,
	eos_token_id: u32
}

unsafe impl Send for CLIPStandardTokenizer {}
unsafe impl Sync for CLIPStandardTokenizer {}

impl CLIPStandardTokenizer {
	/// Loads a CLIP tokenizer from a file.
	pub fn new(path: impl Into<PathBuf>, reconfigure: bool, model_max_length: usize, bos_token_id: u32, eos_token_id: u32) -> anyhow::Result<Self> {
		let path = path.into();
		let bytes = std::fs::read(path)?;
		Self::from_bytes(bytes, reconfigure, model_max_length, bos_token_id, eos_token_id)
	}

	/// Loads a CLIP tokenizer from a byte array.
	pub fn from_bytes<B: AsRef<[u8]>>(bytes: B, reconfigure: bool, model_max_length: usize, bos_token_id: u32, eos_token_id: u32) -> anyhow::Result<Self> {
		let mut tokenizer: Tokenizer = serde_json::from_slice(bytes.as_ref())?;
		// `reconfigure` is disabled in long prompt weighting; LPW has its own padding and truncation strategy that would
		// conflict with this configuration.
		if reconfigure {
			// For some reason, CLIP tokenizers lose their padding and truncation config when converting from the old HF tokenizers
			// format, so we have to add them back here.
			tokenizer
				.with_padding(Some(PaddingParams {
					strategy: PaddingStrategy::Fixed(model_max_length),
					// `clip-vit-base-patch32` and (maybe) all Stable Diffusion models use `"pad_token": "<|endoftext|>"`
					// This info is also lost in translation in HF tokenizers.
					pad_id: eos_token_id,
					..Default::default()
				}))
				.with_truncation(Some(TruncationParams {
					max_length: model_max_length,
					..Default::default()
				}));
		}
		Ok(Self {
			tokenizer,
			model_max_length,
			bos_token_id,
			eos_token_id
		})
	}

	/// Returns the BPE model used by the tokenizer.
	///
	/// # Panics
	///
	/// - Panics with `unreachable!()` if the tokenizer model is not BPE (which would mean this tokenizer isn't a CLIP
	///   tokenizer, which should be impossible).
	#[allow(dead_code)]
	pub fn model(&self) -> &BPE {
		match self.tokenizer.get_model() {
			tokenizers::ModelWrapper::BPE(ref bpe) => bpe,
			_ => unreachable!()
		}
	}

	/// Returns the maximum length of tokens this tokenizer supports. For most CLIP models, this is 77 tokens.
	#[allow(clippy::len_without_is_empty)]
	pub fn len(&self) -> usize {
		self.model_max_length
	}

	/// Returns the ID of the end-of-string token.
	#[allow(dead_code)]
	pub fn eos(&self) -> u32 {
		self.eos_token_id
	}

	/// Returns the ID of the beginning-of-string token.
	#[allow(dead_code)]
	pub fn bos(&self) -> u32 {
		self.bos_token_id
	}

	/// Encodes the input string(s) into arrays of token IDs.
	pub fn encode<'s, 'e, E>(&self, enc: Vec<E>) -> anyhow::Result<Vec<Vec<u32>>>
	where
		E: Into<EncodeInput<'s>> + Send
	{
		Ok(self
			.tokenizer
			.encode_batch(enc, true)
			.map_err(|e| anyhow::anyhow!("{e:?}"))?
			.iter()
			.map(|f| f.get_ids().to_vec())
			.collect())
	}

	/// Encodes the input prompts into an [`Array2`] to be passed to a CLIPTextModel.
	pub fn encode_for_text_model<'s, 'e, E>(&self, enc: Vec<E>) -> anyhow::Result<Array2<i32>>
	where
		E: Into<EncodeInput<'s>> + Send
	{
		let batch_size = enc.len();
		Ok(Array2::from_shape_vec(
			(batch_size, self.len()),
			self.encode(enc)?
				.iter()
				.flat_map(|v| v.iter().map(|tok| *tok as _).collect::<Vec<i32>>())
				.collect()
		)?)
	}
}
