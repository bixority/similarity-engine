use anyhow::Result;
use aws_config;
use aws_sdk_s3::Client;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use serde::{Deserialize, Serialize};
use std::env;
use tokenizers::Tokenizer;
use tokio::io::AsyncReadExt;

#[tokio::main]
async fn main() -> Result<()> {
    // Load AWS configuration (credentials provided by AWS Batch IAM role)
    let config = aws_config::load_from_env().await;
    let s3_client = Client::new(&config);

    // Read S3 bucket and key from environment variables
    let bucket = env::var("S3_BUCKET").expect("S3_BUCKET environment variable not set");
    let key = env::var("S3_KEY").expect("S3_KEY environment variable not set");

    // Download and parse JSON file from S3
    let job_config = download_job_config(&s3_client, &bucket, &key).await?;
    if job_config.job_name != "semantic_scoring" {
        return Err(anyhow::anyhow!("Invalid job_name: expected 'semantic_scoring', got '{}'", job_config.job_name));
    }

    // Load model and tokenizer
    let model_path = env::var("MODEL_PATH").expect("MODEL_PATH environment variable not set");
    let device = if candle_core::utils::cuda_is_available() {
        println!("Using CUDA device");
        Device::new_cuda(0)?
    } else {
        println!("CUDA not available, falling back to CPU");
        Device::Cpu
    };

    // Load tokenizer
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Load model configuration
    let config_path = format!("{}/config.json", model_path);
    let config_content = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
    let model_config: Config = serde_json::from_str(&config_content)
        .map_err(|e| anyhow::anyhow!("Failed to parse config.json: {}", e))?;

    // Load model weights
    let model_weights_path = format!("{}/model.safetensors", model_path);
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_weights_path], DType::F32, &device)? };
    let model = BertModel::load(vb, &model_config)?;

    // Compute semantic scores
    let scores = semantic_scores(&model, &tokenizer, &job_config.value, &job_config.values, &device)?;

    // Prepare results as JSON
    let results: Vec<ScoreResult> = job_config
        .values
        .iter()
        .zip(scores.iter())
        .map(|(text, score)| ScoreResult {
            text: text.clone(),
            score: *score,
        })
        .collect();
    let result_json = serde_json::to_string(&results)?;

    // Upload results to S3 with .result suffix
    let result_key = format!("{}.result", key);
    upload_results(&s3_client, &bucket, &result_key, &result_json).await?;

    // Print results to stdout for logging
    for result in &results {
        println!("Score for '{}': {:.4}", result.text, result.score);
    }

    Ok(())
}

/// Job configuration structure from JSON
#[derive(Deserialize)]
struct JobConfig {
    job_name: String,
    value: String,
    values: Vec<String>,
}

/// Result structure for output JSON
#[derive(Serialize)]
struct ScoreResult {
    text: String,
    score: f32,
}

/// Downloads and parses the JSON configuration file from S3
async fn download_job_config(client: &Client, bucket: &str, key: &str) -> Result<JobConfig> {
    let resp = client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await?;

    let mut data = Vec::new();
    let mut stream = resp.body.into_async_read();
    stream.read_to_end(&mut data).await?;

    let config: JobConfig = serde_json::from_slice(&data)?;
    Ok(config)
}

/// Uploads results to S3
async fn upload_results(client: &Client, bucket: &str, key: &str, data: &str) -> Result<()> {
    client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(data.as_bytes().to_vec().into())
        .content_type("application/json")
        .send()
        .await?;
    println!("Uploaded results to s3://{}/{}", bucket, key);
    Ok(())
}

/// Computes semantic scores using the BERT model and tokenizer
fn semantic_scores(
    model: &BertModel,
    tokenizer: &Tokenizer,
    query: &str,
    candidates: &[String],
    device: &Device,
) -> Result<Vec<f32>> {
    // Prepare texts
    let mut all_texts = vec![query.to_string()];
    all_texts.extend_from_slice(candidates);

    // Tokenize texts
    let encodings = all_texts
        .iter()
        .map(|text| {
            tokenizer
                .encode(text, true)
                .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))
        })
        .collect::<Result<Vec<_>>>()?;

    // Prepare input tensors
    let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(128).min(128);
    let input_ids: Vec<Vec<u32>> = encodings
        .iter()
        .map(|e| {
            let mut ids: Vec<u32> = e.get_ids().iter().map(|&id| id as u32).collect();
            ids.resize(max_len, 0); // Pad with zeros
            ids
        })
        .collect();
    let attention_mask: Vec<Vec<u32>> = encodings
        .iter()
        .map(|e| {
            let mut mask: Vec<u32> = e.get_attention_mask().iter().map(|&m| m as u32).collect();
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

    let dot_products = candidate_embs.matmul(&query_emb.unsqueeze(1)?)?.squeeze(1)?;
    let scores = dot_products.div(&(candidate_norms.mul(&query_norm)?))?;

    // Convert to Vec<f32>
    let scores_vec: Vec<f32> = scores.to_vec1()?;

    Ok(scores_vec)
}