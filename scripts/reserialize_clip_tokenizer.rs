//! Takes in a new/fast-format HF tokenizers configuration file and converts it into CBOR representation for use with
//! pyke Diffusers. This also encodes the tokenizer model's maximum length and beginning- and end-of-string token IDs,
//! as those details are lost or harder to access after the fast format conversion.

use std::path::PathBuf;

use pyke_diffusers::clip::CLIPStandardTokenizerWrapper;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
	let args = std::env::args().collect::<Vec<_>>();
	let in_path = PathBuf::from(&args[1]);
	let out_path = PathBuf::from(&args[2]);
	let model_max_length: usize = args[3].parse()?;
	let bos_token_id: u32 = args[4].parse()?;
	let eos_token_id: u32 = args[5].parse()?;

	let tokenizer_json = std::fs::read(in_path)?;
	let tokenizer: Tokenizer = serde_json::from_slice(&tokenizer_json)?;
	let wrapper = CLIPStandardTokenizerWrapper {
		tokenizer,
		model_max_length,
		bos_token_id,
		eos_token_id
	};
	std::fs::write(out_path, serde_json::to_vec(&wrapper)?)?;

	Ok(())
}
