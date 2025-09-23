# syntax=docker/dockerfile:1.7
# Production-ready multi-stage, multi-arch Dockerfile for Rust semantic scoring service

ARG RUST_VERSION=1.90
ARG DEBIAN_VERSION=trixie

# =============================================================================
# Build stage - Build the Rust application
# =============================================================================
FROM --platform=$BUILDPLATFORM rust:${RUST_VERSION}-${DEBIAN_VERSION} AS builder

# Install build dependencies and cross-compilation tools
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install cross-compilation targets
ARG TARGETPLATFORM
RUN case "$TARGETPLATFORM" in \
    "linux/amd64") echo "x86_64-unknown-linux-gnu" > /target-triple.txt ;; \
    "linux/arm64") echo "aarch64-unknown-linux-gnu" > /target-triple.txt ;; \
    *) echo "Unsupported platform: $TARGETPLATFORM" && exit 1 ;; \
    esac

RUN TARGET_TRIPLE=$(cat /target-triple.txt) && \
    if [ "$TARGET_TRIPLE" != "x86_64-unknown-linux-gnu" ]; then \
        rustup target add $TARGET_TRIPLE; \
        if [ "$TARGET_TRIPLE" = "aarch64-unknown-linux-gnu" ]; then \
            apt-get update && apt-get install -y \
                gcc-aarch64-linux-gnu \
                libc6-dev-arm64-cross \
                && rm -rf /var/lib/apt/lists/*; \
        fi \
    fi

# Configure cross-compilation environment
ENV TARGET_TRIPLE_FILE=/target-triple.txt
RUN TARGET_TRIPLE=$(cat $TARGET_TRIPLE_FILE) && \
    if [ "$TARGET_TRIPLE" = "aarch64-unknown-linux-gnu" ]; then \
        echo "[target.aarch64-unknown-linux-gnu]" >> ~/.cargo/config.toml && \
        echo "linker = \"aarch64-linux-gnu-gcc\"" >> ~/.cargo/config.toml; \
    fi

# Create app user and directory
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser
WORKDIR /app

# Copy dependency files first for better caching
COPY --chown=appuser:appuser Cargo.toml Cargo.lock ./

# Create a dummy main.rs to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies for better layer caching
RUN TARGET_TRIPLE=$(cat /target-triple.txt) && \
    if [ "$TARGET_TRIPLE" != "x86_64-unknown-linux-gnu" ]; then \
        cargo build --release --target=$TARGET_TRIPLE; \
    else \
        cargo build --release; \
    fi && \
    rm -rf src/

# Copy source code
COPY --chown=appuser:appuser src/ ./src/

# Build the actual application
RUN TARGET_TRIPLE=$(cat /target-triple.txt) && \
    if [ "$TARGET_TRIPLE" != "x86_64-unknown-linux-gnu" ]; then \
        cargo build --release --target=$TARGET_TRIPLE && \
        cp target/$TARGET_TRIPLE/release/semantic_scoring /app/semantic_scoring; \
    else \
        cargo build --release && \
        cp target/release/semantic_scoring /app/semantic_scoring; \
    fi

# Strip the binary to reduce size
RUN strip /app/semantic_scoring

# Verify the binary works
RUN /app/semantic_scoring --version 2>/dev/null || echo "Binary built successfully"

# =============================================================================
# Runtime stage - Create minimal production image
# =============================================================================
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04 AS runtime

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r -g 1001 appuser && \
    useradd -r -u 1001 -g appuser -d /app -s /sbin/nologin -c "App User" appuser

# Create directories with proper permissions
RUN mkdir -p /app/models /tmp/app && \
    chown -R appuser:appuser /app /tmp/app && \
    chmod 755 /app && \
    chmod 1777 /tmp/app

# Copy the binary from builder stage
COPY --from=builder --chown=appuser:appuser /app/semantic_scoring /app/semantic_scoring
RUN chmod +x /app/semantic_scoring

# Set working directory
WORKDIR /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
ENV MODEL_PATH=/app/models
ENV TMPDIR=/tmp/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/semantic_scoring --version || exit 1

# Add labels for better maintainability
LABEL maintainer="bixority@gmail.com" \
      org.opencontainers.image.title="Semantic Scoring Service" \
      org.opencontainers.image.description="BERT-based semantic similarity scoring service" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="Bixority SIA" \
      org.opencontainers.image.licenses="GPL-3.0" \
      org.opencontainers.image.source="https://github.com/bixority/similarity-engine"

# Expose port if needed (uncomment if your app serves HTTP)
# EXPOSE 8080

# Default command
ENTRYPOINT ["/app/semantic_scoring"]
CMD []