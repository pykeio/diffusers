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

use std::num::ParseFloatError;

use ndarray::{s, Array2, Array3, Axis, NewAxis};
use ort::{
	tensor::{FromArray, InputTensor},
	OrtResult, Session
};
use regex::Regex;

use crate::{text_embeddings::TextEmbeddings, Prompt};

lazy_static::lazy_static! {
	static ref RE_ATTENTION: Regex = Regex::new(
		r"(?x)
	\\\(|
	\\\)|
	\\\[|
	\\]|
	\\\\|
	\\|
	\(|
	\[|
	:([+-]?[.\d]+)\)|
	\)|
	]|
	[^\\()\[\]:]+|
	:
	"
	)
	.unwrap();
}

type LpwTokens = Vec<Vec<u32>>;
type LpwWeights = Vec<Vec<f32>>;

fn parse_prompt_attention(text: impl AsRef<str>) -> Result<Vec<(String, f32)>, ParseFloatError> {
	let mut res: Vec<(String, f32)> = Vec::new();
	let mut round_brackets = Vec::new();
	let mut square_brackets = Vec::new();

	const ROUND_BRACKET_MULTIPLIER: f32 = 1.1;
	const SQUARE_BRACKET_MULTIPLIER: f32 = 1.0 / 1.1;

	for m in RE_ATTENTION.captures_iter(text.as_ref()) {
		let text = m.get(0).map_or("", |m| m.as_str());
		let weight = m.get(1).map(|m| m.as_str());

		let rlen = res.len();

		if let Some(stripped) = text.strip_prefix('\\') {
			res.push((stripped.to_owned(), 1.0));
		} else if text == "(" {
			round_brackets.push(rlen);
		} else if text == "[" {
			square_brackets.push(rlen);
		} else if weight.is_some() && !round_brackets.is_empty() {
			let multiplier = weight.unwrap().parse::<f32>()?;
			for i in res.iter_mut().take(rlen).skip(round_brackets.pop().unwrap()) {
				i.1 *= multiplier;
			}
		} else if text == ")" && !round_brackets.is_empty() {
			for i in res.iter_mut().take(rlen).skip(round_brackets.pop().unwrap()) {
				i.1 *= ROUND_BRACKET_MULTIPLIER;
			}
		} else if text == ")" && !square_brackets.is_empty() {
			for i in res.iter_mut().take(rlen).skip(square_brackets.pop().unwrap()) {
				i.1 *= SQUARE_BRACKET_MULTIPLIER;
			}
		} else {
			res.push((text.to_owned(), 1.0));
		}
	}

	// process remaining
	let rlen = res.len();
	for pos in round_brackets.iter() {
		for i in res.iter_mut().take(rlen).skip(*pos) {
			i.1 *= ROUND_BRACKET_MULTIPLIER;
		}
	}
	for pos in square_brackets.iter() {
		for i in res.iter_mut().take(rlen).skip(*pos) {
			i.1 *= SQUARE_BRACKET_MULTIPLIER;
		}
	}

	if rlen == 0 {
		res = vec![("".to_owned(), 1.0)];
	}

	let mut i = 0;
	while i + 1 < res.len() {
		let next = res[i + 1].clone();
		if res[i].1 == next.1 {
			res[i].0 += &next.0;
			res.remove(i + 1);
		} else {
			i += 1;
		}
	}

	Ok(res)
}

fn get_prompts_with_weights(embeddings: &TextEmbeddings, prompts: Prompt, max_length: usize) -> anyhow::Result<(LpwTokens, LpwWeights)> {
	let mut tokens = vec![];
	let mut weights = vec![];
	for prompt in prompts.iter() {
		let texts_and_weights = parse_prompt_attention(prompt)?;
		let mut text_token = vec![];
		let mut text_weight = vec![];
		for (word, weight) in texts_and_weights {
			let token = &embeddings.tokenizer.encode(vec![word])?[0];
			let token = &token[1..token.len() - 1];
			text_token.extend_from_slice(token);
			text_weight.extend_from_slice(&[weight].repeat(token.len()));
			if text_token.len() > max_length {
				break;
			}
		}

		if text_token.len() > max_length {
			text_token = text_token[..max_length].to_vec();
			text_weight = text_weight[..max_length].to_vec();
		}

		tokens.push(text_token);
		weights.push(text_weight);
	}
	Ok((tokens, weights))
}

fn pad_tokens_and_weights(
	mut tokens: LpwTokens,
	mut weights: LpwWeights,
	max_length: usize,
	bos_id: u32,
	eos_id: u32,
	no_boseos_middle: bool,
	chunk_length: usize
) -> (LpwTokens, LpwWeights) {
	let max_embeddings_multiples = (max_length - 2) / (chunk_length - 2);
	let weights_length = if no_boseos_middle { max_length } else { max_embeddings_multiples * chunk_length };
	for i in 0..tokens.len() {
		let mut nvt = vec![bos_id];
		nvt.extend_from_slice(&tokens[i]);
		nvt.extend_from_slice(&[eos_id].repeat(max_length - 1 - tokens[i].len()));
		tokens[i] = nvt;

		if no_boseos_middle {
			let mut nvw = vec![1.0];
			nvw.extend_from_slice(&weights[i]);
			nvw.extend_from_slice(&[1.0].repeat(max_length - 1 - weights[i].len()));
			weights[i] = nvw;
		} else {
			let mut w = vec![];
			if weights[i].is_empty() {
				w.extend_from_slice(&[1.0].repeat(weights_length));
			} else {
				for j in 0..max_embeddings_multiples {
					w.push(1.0);
					w.extend_from_slice(&weights[i][j * (chunk_length - 2)..weights[i].len().min((j + 1) * (chunk_length - 2))]);
					w.push(1.0);
				}
				w.extend_from_slice(&[1.0].repeat(weights_length - w.len()));
			}
			weights[i] = w;
		}
	}
	(tokens, weights)
}

pub fn get_unweighted_text_embeddings(
	#[cfg_attr(test, allow(unused))] embeddings: &TextEmbeddings,
	text_encoder: &Session,
	text_input: Array2<i32>,
	chunk_length: usize,
	no_boseos_middle: bool
) -> OrtResult<Array3<f32>> {
	let max_embeddings_multiples = (text_input.shape()[1] - 2) / (chunk_length - 2);
	if max_embeddings_multiples > 1 {
		let mut text_embeddings = Vec::new();
		for i in 0..max_embeddings_multiples {
			let mut text_input_chunk = text_input
				.slice(s![.., (i * (chunk_length - 2))..((i + 1) * (chunk_length - 2) + 2)])
				.to_owned();

			text_input_chunk.slice_mut(s![.., 0]).assign(&text_input.slice(s![0, 0]));
			text_input_chunk.slice_mut(s![.., -1]).assign(&text_input.slice(s![0, -1]));

			let text_input_chunk = if embeddings.is_empty() {
				// no external embeds
				InputTensor::from_array(text_input_chunk.into_dyn())
			} else {
				// pre-embed
				let text_input_chunk = text_input_chunk.into_raw_vec();
				InputTensor::from_array(embeddings.embed(text_input_chunk.iter().map(|f| *f as u32).collect()).into_dyn())
			};

			let chunk_embeddings = text_encoder.run(vec![text_input_chunk])?;
			let chunk_embeddings: Array3<f32> = chunk_embeddings[0].try_extract()?.view().to_owned().into_dimensionality().unwrap();

			#[allow(clippy::reversed_empty_ranges)]
			let view = if no_boseos_middle {
				if i == 0 {
					chunk_embeddings.slice(s![.., ..-1, ..])
				} else if i == max_embeddings_multiples - 1 {
					chunk_embeddings.slice(s![.., 1.., ..])
				} else {
					chunk_embeddings.slice(s![.., 1..-1, ..])
				}
			} else {
				chunk_embeddings.view()
			};

			text_embeddings.push(view.to_owned());
		}

		let mut x1 = text_embeddings[0].to_owned();
		for x in &text_embeddings[1..] {
			x1.append(Axis(1), x.view()).unwrap();
		}
		Ok(x1)
	} else {
		let text_input = if embeddings.is_empty() {
			// no external embeds
			InputTensor::from_array(text_input.into_dyn())
		} else {
			// pre-embed
			let text_input = text_input.into_raw_vec();
			InputTensor::from_array(embeddings.embed(text_input.iter().map(|f| *f as u32).collect()).into_dyn())
		};

		let text_embeddings = text_encoder.run(vec![text_input])?;
		Ok(text_embeddings[0].try_extract()?.view().to_owned().into_dimensionality().unwrap())
	}
}

pub fn get_weighted_text_embeddings(
	embeddings: &TextEmbeddings,
	text_encoder: &Session,
	prompt: Prompt,
	neg_prompt: Option<Prompt>,
	max_embeddings_multiples: usize,
	no_boseos_middle: bool
) -> anyhow::Result<(Array3<f32>, Option<Array3<f32>>)> {
	let max_length = (embeddings.tokenizer.len() - 2) * max_embeddings_multiples + 2;

	let (prompt_tokens, prompt_weights) = get_prompts_with_weights(embeddings, prompt, max_length - 2)?;
	let uncond_ptt = if let Some(neg_prompt) = neg_prompt {
		Some(get_prompts_with_weights(embeddings, neg_prompt, max_length - 2)?)
	} else {
		None
	};

	let mut max_length = prompt_tokens.iter().map(|t| t.len()).max().unwrap_or(0);
	if let Some((uncond_tokens, _)) = uncond_ptt.as_ref() {
		max_length = max_length.max(uncond_tokens.iter().map(|t| t.len()).max().unwrap_or(0));
	}

	let max_embeddings_multiples = max_embeddings_multiples.min((max_length - 1) / (embeddings.tokenizer.len() - 2) + 1);
	let max_embeddings_multiples = 1.max(max_embeddings_multiples);
	let max_length = (embeddings.tokenizer.len() - 2) * max_embeddings_multiples + 2;

	let bos_id = embeddings.tokenizer.bos();
	let eos_id = embeddings.tokenizer.eos();
	let (prompt_tokens, prompt_weights) =
		pad_tokens_and_weights(prompt_tokens, prompt_weights, max_length, bos_id, eos_id, no_boseos_middle, embeddings.tokenizer.len());
	let uncond_padded = if let Some((uncond_tokens, uncond_weights)) = uncond_ptt {
		Some(pad_tokens_and_weights(uncond_tokens, uncond_weights, max_length, bos_id, eos_id, no_boseos_middle, embeddings.tokenizer.len()))
	} else {
		None
	};

	let text_embeddings = get_unweighted_text_embeddings(
		embeddings,
		text_encoder,
		Array2::from_shape_vec((prompt_tokens.len(), prompt_tokens[0].len()), prompt_tokens.concat())?.map(|f| *f as i32),
		embeddings.tokenizer.len(),
		no_boseos_middle
	)?;

	let previous_mean = text_embeddings.mean_axis(Axis(2)).unwrap().mean_axis(Axis(1)).unwrap();
	let text_embeddings = text_embeddings
		* Array2::from_shape_vec((prompt_weights.len(), prompt_weights[0].len()), prompt_weights.concat())?
			.slice(s![.., .., NewAxis])
			.to_owned();
	let current_mean = text_embeddings.mean_axis(Axis(2)).unwrap().mean_axis(Axis(1)).unwrap();
	let text_embeddings = text_embeddings * (previous_mean / current_mean);

	let uncond_embeddings = if let Some((uncond_tokens, uncond_weights)) = uncond_padded {
		let uncond_embeddings = get_unweighted_text_embeddings(
			embeddings,
			text_encoder,
			Array2::from_shape_vec((uncond_tokens.len(), uncond_tokens[0].len()), uncond_tokens.concat())?.map(|f| *f as i32),
			embeddings.tokenizer.len(),
			no_boseos_middle
		)?;
		let previous_mean = uncond_embeddings.mean_axis(Axis(2)).unwrap().mean_axis(Axis(1)).unwrap();
		let uncond_embeddings = uncond_embeddings
			* Array2::from_shape_vec((uncond_weights.len(), uncond_weights[0].len()), uncond_weights.concat())?
				.slice(s![.., .., NewAxis])
				.to_owned();
		let current_mean = uncond_embeddings.mean_axis(Axis(2)).unwrap().mean_axis(Axis(1)).unwrap();
		Some(uncond_embeddings * (previous_mean / current_mean))
	} else {
		None
	};

	Ok((text_embeddings, uncond_embeddings))
}
