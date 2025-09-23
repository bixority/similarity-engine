use aws_config;
use aws_sdk_s3::Client;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;

/// Job configuration structure from JSON
#[derive(Deserialize)]
pub struct JobConfig {
    pub job_name: String,
    pub value: String,
    pub values: Vec<String>,
}

/// Result structure for output JSON
#[derive(Serialize)]
pub struct ScoreResult {
    pub text: String,
    pub score: f32,
}

/// Downloads and parses the JSON configuration file from S3
pub async fn download_job_config(
    client: &Client,
    bucket: &str,
    key: &str,
) -> anyhow::Result<JobConfig> {
    let resp = client.get_object().bucket(bucket).key(key).send().await?;

    let mut data = Vec::new();
    let mut stream = resp.body.into_async_read();
    stream.read_to_end(&mut data).await?;

    let config: JobConfig = serde_json::from_slice(&data)?;
    Ok(config)
}

/// Uploads results to S3
pub async fn upload_results(
    client: &Client,
    bucket: &str,
    key: &str,
    data: &str,
) -> anyhow::Result<()> {
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
