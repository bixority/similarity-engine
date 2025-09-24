use candle_core::{DType, Device, Tensor};
use candle_transformers::models::bert::BertModel;
use tokenizers::Tokenizer;

/// Computes semantic scores using the BERT model and tokenizer
pub fn semantic_scores(
    model: &BertModel,
    tokenizer: &Tokenizer,
    query: &str,
    candidates: &[String],
    device: &Device,
) -> anyhow::Result<Vec<f32>> {
    // Prepare texts
    let mut all_texts = vec![query.to_string()];
    all_texts.extend_from_slice(candidates);

    // Tokenize texts
    let encodings = all_texts
        .iter()
        .map(|text| {
            tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))
        })
        .collect::<Result<Vec<_>, anyhow::Error>>()?;

    // Prepare input tensors
    let max_len = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(128)
        .min(128);
    let input_ids: Vec<Vec<u32>> = encodings
        .iter()
        .map(|e| {
            let mut ids: Vec<u32> = e.get_ids().to_vec();
            ids.resize(max_len, 0); // Pad with zeros
            ids
        })
        .collect();
    let attention_mask: Vec<Vec<u32>> = encodings
        .iter()
        .map(|e| {
            let mut mask: Vec<u32> = e.get_attention_mask().to_vec();
            mask.resize(max_len, 0); // Pad with zeros
            mask
        })
        .collect();

    // Convert to tensors
    let input_ids = Tensor::new(input_ids, device)?.to_dtype(DType::U32)?;
    let attention_mask = Tensor::new(attention_mask, device)?.to_dtype(DType::U32)?;

    // Create token_type_ids (all zeros for single sequence)
    let token_type_ids = Tensor::zeros((input_ids.dim(0)?, input_ids.dim(1)?), DType::U32, device)?;

    // Run model inference
    let embeddings = model.forward(&input_ids, &attention_mask, Some(&token_type_ids))?;

    // Mean pooling (sentence-transformers style)
    let attention_mask_f32 = attention_mask.to_dtype(DType::F32)?;
    let mask_expanded = attention_mask_f32.unsqueeze(2)?;
    let masked_embeddings = embeddings.mul(&mask_expanded)?;
    let sum_embeddings = masked_embeddings.sum(1)?;
    let mask_sum = mask_expanded.sum(1)?.clamp(1.0, f64::INFINITY)?;
    let pooled_embeddings = sum_embeddings.div(&mask_sum)?;

    // Split query and candidates
    let query_emb = pooled_embeddings.get(0)?; // Shape [hidden_size]
    let candidate_embs = pooled_embeddings.narrow(0, 1, candidates.len())?; // Shape [n_candidates, hidden_size]

    // Compute cosine similarities
    let query_norm = query_emb.sqr()?.sum_all()?.sqrt()?;
    let candidate_norms = candidate_embs.sqr()?.sum(1)?.sqrt()?;

    let dot_products = candidate_embs
        .matmul(&query_emb.unsqueeze(1)?)?
        .squeeze(1)?;
    let scores = dot_products.div(&(candidate_norms.mul(&query_norm)?))?;

    // Convert to Vec<f32>
    let scores_vec: Vec<f32> = scores.to_vec1()?;

    Ok(scores_vec)
}
