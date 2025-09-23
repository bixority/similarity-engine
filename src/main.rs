mod jobs;
mod scoring;

use crate::jobs::{download_job_config, upload_results, ScoreResult};
use crate::scoring::semantic_scores;
use anyhow::Result;
use aws_config;
use aws_sdk_s3::Client;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use std::env;
use tokenizers::Tokenizer;

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
