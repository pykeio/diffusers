use std::{
	collections::HashMap,
	fs::File,
	io::{self, BufRead, BufReader},
	path::{Path, PathBuf}
};

use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::{concatenate, stack, Array2, Array3, Axis};

use crate::clip::CLIPStandardTokenizer;

pub struct AddedToken {
	pub tok: String,
	pub tid: u32
}

pub struct TextEmbeddings {
	pub tokenizer: CLIPStandardTokenizer,
	text_hidden_size: u32,
	pub tokens: HashMap<u32, Array2<f32>>
}

impl TextEmbeddings {
	pub fn from_file<P: AsRef<Path>>(path: P, tokenizer: CLIPStandardTokenizer) -> io::Result<Self> {
		let path = path.as_ref();
		Self::from_reader(BufReader::new(File::open(path)?), tokenizer)
	}

	pub fn from_reader<R: BufRead>(mut reader: R, tokenizer: CLIPStandardTokenizer) -> io::Result<Self> {
		let n_tokens = reader.read_u32::<LittleEndian>()?;
		let text_hidden_size = reader.read_u32::<LittleEndian>()?;

		if n_tokens != 0 {
			assert_eq!(n_tokens as usize, tokenizer.inner.get_vocab_size(true));
		}

		let mut tokens = HashMap::with_capacity(n_tokens as _);
		for i in 0..n_tokens {
			let mut token = Vec::with_capacity(text_hidden_size as _);
			for _ in 0..text_hidden_size {
				token.push(reader.read_f32::<LittleEndian>()?);
			}

			let token = Array2::from_shape_vec((1, text_hidden_size as _), token).unwrap();
			tokens.insert(i, token);
		}

		Ok(Self { tokenizer, text_hidden_size, tokens })
	}

	pub fn add_token_from_file<P: AsRef<Path>>(&mut self, path: P) -> io::Result<AddedToken> {
		let path = path.as_ref();
		self.add_token_from_reader(BufReader::new(File::open(path)?))
	}

	pub fn add_token_from_reader<R: BufRead>(&mut self, mut reader: R) -> io::Result<AddedToken> {
		let n_vectors = reader.read_u32::<LittleEndian>()?;
		let text_hidden_size = reader.read_u32::<LittleEndian>()?;
		assert_eq!(text_hidden_size, self.text_hidden_size);

		let name_len = reader.read_u32::<LittleEndian>()?;
		let mut name_buf = vec![0; name_len as _];
		reader.read_exact(&mut name_buf)?;
		let token_name = String::from_utf8_lossy(&name_buf).to_string();

		let mut buf = Vec::with_capacity((n_vectors * text_hidden_size) as usize);
		for _ in 0..(n_vectors * text_hidden_size) {
			buf.push(reader.read_f32::<LittleEndian>()?);
		}

		let embeds = Array2::from_shape_vec((n_vectors as usize, text_hidden_size as usize), buf).unwrap();
		Ok(self.add_token(token_name, embeds))
	}

	pub fn add_token(&mut self, tok: String, embeds: Array2<f32>) -> AddedToken {
		self.tokenizer.inner.add_tokens(&[tokenizers::AddedToken {
			content: tok.clone(),
			single_word: true,
			special: false,
			lstrip: false,
			rstrip: false,
			normalized: false
		}]);
		let token_id = self.tokenizer.inner.token_to_id(&tok).unwrap();

		self.tokens.insert(token_id, embeds);

		AddedToken { tok, tid: token_id }
	}

	pub fn embed(&self, token_ids: Array2<i32>) -> Array3<f32> {
		let (batch_size, max_len) = (token_ids.shape()[0], token_ids.shape()[1]);
		let mut embeds = Vec::with_capacity(batch_size);
		for batch in token_ids.axis_iter(Axis(0)) {
			let mut batch_embeds = Vec::with_capacity(max_len);
			for tok_id in batch {
				let tok = self.tokens.get(&(*tok_id as u32)).unwrap();
				batch_embeds.push(tok.view());
			}
			embeds.push(concatenate(Axis(0), &batch_embeds).unwrap());
		}

		stack(Axis(0), &embeds.iter().map(|f| f.view()).collect::<Vec<_>>()).unwrap()
	}

	pub fn len(&self) -> usize {
		self.tokens.len()
	}

	pub fn is_empty(&self) -> bool {
		self.tokens.len() == 0
	}
}
