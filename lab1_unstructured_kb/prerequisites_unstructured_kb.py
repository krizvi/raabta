#!/usr/bin/env python
# coding: utf-8

"""
1.1 – Create an unstructured Amazon Bedrock Knowledge Base.

This script does what the original notebook did:
- Sets up AWS clients and logging
- Creates a unique S3 bucket
- Uploads sample unstructured documents
- Creates an Amazon Bedrock Knowledge Base (using the helper class)
- Starts an ingestion job and prints the Knowledge Base ID
- Stores the KB ID in AWS Systems Manager Parameter Store

You can run this as a standalone script once your AWS credentials and region
are configured (e.g., via environment variables or AWS config).
"""

import json
import logging
import os
import random
import string
import time
from datetime import datetime

import boto3
import botocore

# -------------------------------------------------------------------------
# AWS client setup and logging
# -------------------------------------------------------------------------

# Session and region resolution based on your current AWS config
session = boto3.session.Session()
region = session.region_name
if region is None:
    raise RuntimeError(
        "No AWS region set. Configure AWS_REGION or default region first."
    )

# Basic STS call just to show which account we are using
sts_client = boto3.client("sts", region_name=region)
account_id = sts_client.get_caller_identity()["Account"]

# Core service clients used in this script
s3_client = boto3.client("s3", region_name=region)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
bedrock_client = boto3.client("bedrock", region_name=region)

# Configure human-readable logging for debugging
logging.basicConfig(
    format="[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

print(f"AWS region: {region}")
print(f"AWS account ID: {account_id}")

# -------------------------------------------------------------------------
# Helper imports – workshop helper for Knowledge Base management
# -------------------------------------------------------------------------

# If this script is run from inside `lab1/` directory, move one level up so
# that `utils.knowledge_base` can be imported consistently.
if "lab1" in os.getcwd():
    os.chdir("..")
else:
    # Just show where we are running from for visibility
    print(f"Current working directory: {os.getcwd()}")

from utils.knowledge_base import BedrockKnowledgeBase  # noqa: E402

# -------------------------------------------------------------------------
# Resource naming and configuration
# -------------------------------------------------------------------------

# Generate a short random suffix to avoid S3 / KB naming collisions
suffix = f"krizvi-{"".join(random.choices(string.ascii_lowercase + string.digits, k=8))}"
print(f"Suffix for resources: {suffix}")

# Knowledge base identity and models
knowledge_base_name = f"product-reviews-unstructured-kb-{suffix}"
knowledge_base_description = (
    "Unstructured Knowledge Base containing product review documents."
)

# Foundation + embedding models for this workshop
# We use an inference profile name for generation (global.<modelId>)
foundation_model = "anthropic.claude-haiku-4-5-20251001-v1:0"
generation_model = "global." + foundation_model
embedding_model = "cohere.embed-multilingual-v3"

# S3 bucket and data-source definition for the knowledge base
data_bucket_name = f"product-reviews-unstructured-{suffix}-bucket"
data_sources = [{"type": "S3", "bucket_name": data_bucket_name}]

# -------------------------------------------------------------------------
# Model sanity checks (optional but useful in a workshop)
# -------------------------------------------------------------------------


def verify_bedrock_models():
    """
    Make quick test calls to:
    - list foundation models (to verify marketplace access)
    - invoke the inference profile for the foundation model
    - invoke the embedding model

    This is meant as a sanity check before we create the knowledge base.
    """
    # 1) Verify that Cohere / Claude models are visible
    try:
        models = bedrock_client.list_foundation_models()
        cohere_models = [
            m for m in models["modelSummaries"] if "cohere" in m["modelId"].lower()
        ]
        claude_models = [
            m for m in models["modelSummaries"] if "claude" in m["modelId"].lower()
        ]

        print(
            f"✅ Bedrock access verified - found {len(cohere_models)} Cohere models available"
        )
        if claude_models:
            print(
                f"   Example Claude models: {[m['modelId'] for m in claude_models[:3]]}"
            )
        if cohere_models:
            print(
                f"   Example Cohere models: {[m['modelId'] for m in cohere_models[:3]]}"
            )
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  Bedrock model access check: {e}")

    # 2) Test foundation model using the global inference profile
    try:
        response = bedrock_runtime.invoke_model(
            modelId=generation_model,
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            ),
            contentType="application/json",
        )
        print(f"Foundation model {foundation_model} is active (test invoke succeeded).")
    except Exception as e:  # noqa: BLE001
        print(f"Foundation model error: {e}")

    # 3) Test embedding model with a minimal request
    try:
        response = bedrock_runtime.invoke_model(
            modelId=embedding_model,
            body=json.dumps(
                {
                    "texts": ["test"],
                    "input_type": "search_document",
                }
            ),
            contentType="application/json",
        )
        print(f"Embedding model {embedding_model} is active (test invoke succeeded).")
    except Exception as e:  # noqa: BLE001
        print(f"Embedding model error: {e}")


# -------------------------------------------------------------------------
# S3 helpers – create bucket and upload local sample documents
# -------------------------------------------------------------------------


def create_s3_bucket(bucket_name: str, region_name: str | None = None) -> None:
    """
    Create an S3 bucket in the given region (handles us-east-1 separately).

    If the bucket already exists and is owned by you, the function just prints a note.
    """
    s3 = boto3.client("s3", region_name=region_name)

    try:
        if region_name is None or region_name == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region_name},
            )
        print(f"Bucket '{bucket_name}' created successfully.")
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "BucketAlreadyOwnedByYou":
            print(f"Bucket '{bucket_name}' already exists and is owned by you.")
        else:
            print(
                f"Failed to create bucket: {e.response['Error'].get('Message', str(e))}"
            )


def upload_directory(path: str, bucket_name: str) -> None:
    """
    Upload all files from a local directory tree into an S3 bucket.

    Each filename (not full path) is used as the S3 object key.
    """
    file_count = 0
    for root, _, files in os.walk(path):
        for filename in files:
            file_to_upload = os.path.join(root, filename)
            print(f"Uploading file {file_to_upload} to {bucket_name}")
            s3_client.upload_file(file_to_upload, bucket_name, filename)
            file_count += 1

    if file_count == 0:
        raise ValueError(f"No files found in {path}")

    print(f"Successfully uploaded {file_count} files to {bucket_name}")


# -------------------------------------------------------------------------
# Knowledge Base creation and ingestion
# -------------------------------------------------------------------------


def main() -> None:
    """
    Main orchestration:
    - verify models
    - create S3 bucket and upload data
    - create the unstructured KB
    - start ingestion, print KB ID
    - store KB ID in SSM
    """
    print("=" * 60)
    print("Verifying Bedrock models (foundation + embedding)...")
    verify_bedrock_models()

    print("=" * 60)
    print(f"Creating S3 bucket for unstructured data: {data_bucket_name}")
    create_s3_bucket(data_bucket_name, region)

    # The notebook expected this folder to exist relative to the repo root
    sample_data_dir = "sample_unstructured_data/selected_reviews"
    print(f"Uploading sample documents from: {sample_data_dir}")
    upload_directory(sample_data_dir, data_bucket_name)

    print("=" * 60)
    print("Creating Amazon Bedrock Knowledge Base (unstructured)...")

    # Instantiate the helper object that wraps Bedrock KB APIs
    unstructured_knowledge_base = BedrockKnowledgeBase(
        kb_name=knowledge_base_name,
        kb_description=knowledge_base_description,
        generation_model=generation_model,
        data_sources=data_sources,
        embedding_model=embedding_model,
        chunking_strategy="FIXED_SIZE",
        suffix=f"{suffix}-u",
    )

    # Give the KB a bit of time to finish creating its data source before starting ingestion
    print("Waiting 60 seconds for the knowledge base data source to become ready...")
    time.sleep(60)

    print("Starting ingestion job for unstructured knowledge base...")
    unstructured_knowledge_base.start_ingestion_job()

    # Capture and print the KB ID so it can be used in later steps
    unstructured_kb_id = unstructured_knowledge_base.get_knowledge_base_id()

    kb_region = region  # explicit for clarity in later scripts

    print("=" * 60)
    print(f"Unstructured Knowledge Base ID: {unstructured_kb_id}")
    print(f"Region: {kb_region}")
    print(f"S3 Bucket: {data_bucket_name}")
    print("=" * 60)
    print("Configuration created successfully!")

    # Store KB ID in SSM Parameter Store to be discoverable from other scripts/notebooks
    param_name = "/app/intelligent_rag/agentcore/unstructured_kb_id"
    ssm = boto3.client("ssm", region_name=region)
    ssm.put_parameter(
        Name=param_name,
        Value=unstructured_kb_id,
        Type="String",
        Overwrite=True,
    )
    print(f"Stored {unstructured_kb_id} in SSM parameter: {param_name}")


if __name__ == "__main__":
    main()
