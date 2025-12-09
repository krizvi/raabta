#!/usr/bin/env python
# coding: utf-8

"""
1.2 – Test the unstructured Amazon Bedrock Knowledge Base.

This script:
- Loads the unstructured KB configuration (ID, region, bucket)
- Configures foundation and metadata-filter models
- Calls Bedrock Agent Runtime `retrieve_and_generate` against the KB
- Optionally enables *implicit* metadata filtering on attributes like rating, product_type
- Pretty-prints the answer, citations, and metadata for inspection
"""

import os
import json
import time
from datetime import datetime

import boto3
import logging

# -------------------------------------------------------------------------
# AWS clients and basic logging
# -------------------------------------------------------------------------

session = boto3.session.Session()
region = session.region_name
if region is None:
    raise RuntimeError(
        "No AWS region set. Configure AWS_REGION or default region first."
    )

sts_client = boto3.client("sts", region_name=region)
account_id = sts_client.get_caller_identity()["Account"]

bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime", region_name=region)
ssm_client = boto3.client("ssm", region_name=region)

logging.basicConfig(
    format="[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

print(f"AWS Region: {region}")
print(f"AWS Account ID: {account_id}")

# -------------------------------------------------------------------------
# Load Knowledge Base configuration
# The original notebook pulled these from Jupyter's %store;
# here we use (in order of preference):
#   1. Environment variables
#   2. SSM Parameter Store for the KB ID
# -------------------------------------------------------------------------


def load_unstructured_kb_config():
    """Resolve KB ID, region, and bucket name from env vars / SSM."""
    # KB ID: environment override first, then SSM
    kb_id = os.getenv("UNSTRUCTURED_KB_ID")
    if not kb_id:
        try:
            resp = ssm_client.get_parameter(
                Name="/app/intelligent_rag/agentcore/unstructured_kb_id"
            )
            kb_id = resp["Parameter"]["Value"]
        except ssm_client.exceptions.ParameterNotFound:
            raise RuntimeError(
                "Unstructured KB ID not found. "
                "Set UNSTRUCTURED_KB_ID env var or create the KB with 1.1 first."
            )

    # Region: default to current region; allow override
    kb_region = os.getenv("KB_REGION", region)

    # Bucket is just informational for this script; best-effort resolution
    data_bucket_name = os.getenv("DATA_BUCKET_NAME")
    if not data_bucket_name:
        try:
            resp = ssm_client.get_parameter(
                Name="/app/intelligent_rag/agentcore/data_bucket_name"
            )
            data_bucket_name = resp["Parameter"]["Value"]
        except ssm_client.exceptions.ParameterNotFound:
            data_bucket_name = "<unknown / not stored in SSM>"

    print("Unstructured Knowledge Base Configuration:")
    print("=" * 50)
    print(f"KB ID: {kb_id}")
    print(f"Region: {kb_region}")
    print(f"S3 Bucket: {data_bucket_name}")
    print("=" * 50)

    return kb_id, kb_region, data_bucket_name


unstructured_kb_id, kb_region, data_bucket_name = load_unstructured_kb_config()

# -------------------------------------------------------------------------
# Test configuration for models and inference profiles
# -------------------------------------------------------------------------

# The underlying model ID (no "global." prefix)
foundation_model = "anthropic.claude-haiku-4-5-20251001-v1:0"

# For metadata filtering, we also use Haiku 4.5
metadata_filter_model = "anthropic.claude-haiku-4-5-20251001-v1:0"

# Number of documents to retrieve per query
max_results = 5

# Global inference profile ARN for generation (uses "global.<modelId>")
foundation_model_arn = (
    f"arn:aws:bedrock:{region}:{account_id}:inference-profile/global.{foundation_model}"
)

# Global inference profile ARN used by implicit metadata filtering
metadata_filter_model_arn = f"arn:aws:bedrock:{kb_region}:{account_id}:inference-profile/global.{metadata_filter_model}"

print(f"Using foundation model: {foundation_model}")
print(f"Foundation model ARN: {foundation_model_arn}")
print(f"Metadata filter model: {metadata_filter_model}")
print(f"Metadata filter model ARN: {metadata_filter_model_arn}")
print(f"Max results per query: {max_results}")
print(
    "\n📖 See also: Knowledge Base test configuration and implicit filtering in the AWS docs."
)

# -------------------------------------------------------------------------
# Core helper: run a query against the KB (with optional metadata filtering)
# -------------------------------------------------------------------------


def run_unstructured_kb_query(query: str, use_metadata_filtering: bool = False) -> None:
    """
    Call Bedrock Agent Runtime retrieve_and_generate against the unstructured KB.

    :param query: Natural-language question to ask
    :param use_metadata_filtering: If True, enable implicit metadata filtering based on
                                   document attributes like product_type, rating, etc.
    """
    try:
        # Base retrieval configuration: vector search only
        retrieval_config = {
            "vectorSearchConfiguration": {
                "numberOfResults": max_results,
            }
        }

        # If we opt in, attach implicit metadata filter configuration
        if use_metadata_filtering:
            retrieval_config["vectorSearchConfiguration"][
                "implicitFilterConfiguration"
            ] = {
                "metadataAttributes": [
                    {
                        "key": "product_type",
                        "type": "STRING",
                        # Short, human description to help the model learn how to use this field
                        "description": (
                            "The type of product being reviewed, such as 'cookbook', "
                            "'speaker', 'educational toy', 'board game', 'shirt', "
                            "'self-help', 'furniture', etc."
                        ),
                    },
                    {
                        "key": "rating",
                        "type": "NUMBER",
                        "description": "The rating given by the customer, ranging from 1 to 5 stars.",
                    },
                    {
                        "key": "created_at",
                        "type": "STRING",
                        "description": "The date when the review was created in YYYY-MM-DD format.",
                    },
                    {
                        "key": "product_id",
                        "type": "STRING",
                        "description": "The unique identifier of the product being reviewed.",
                    },
                    {
                        "key": "customer_id",
                        "type": "STRING",
                        "description": "The unique identifier of the customer who wrote the review.",
                    },
                ],
                # The model used to infer metadata filters from the query
                "modelArn": metadata_filter_model_arn,
            }

        filter_status = (
            "WITH METADATA FILTERING"
            if use_metadata_filtering
            else "WITHOUT METADATA FILTERING"
        )
        print(f"\n{'=' * 20} RETRIEVE AND GENERATE API {filter_status} {'=' * 20}")

        # Prompt template used for generation; $search_results$ and $output_format_instructions$
        # are filled by the KB service, and $query$ is replaced with the user's question.
        prompt_template = (
            "You are an assistant helping a product team understand customer feedback.\n\n"
            "Use ONLY the following search results:\n"
            "$search_results$\n\n"
            "$output_format_instructions$\n\n"
            "Answer the customer's question:\n"
            "$query$"
        )

        # Call the API
        rag_response = bedrock_agent_runtime_client.retrieve_and_generate(
            input={"text": query},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": unstructured_kb_id,
                    "modelArn": foundation_model_arn,
                    "retrievalConfiguration": retrieval_config,
                    "generationConfiguration": {
                        "promptTemplate": {
                            "textPromptTemplate": prompt_template,
                        }
                    },
                },
            },
        )

        # Pretty-print the result and citations
        display_rag_results_with_metadata(
            rag_response,
            query=query,
            metadata_filtering_used=use_metadata_filtering,
        )

    except Exception as e:  # noqa: BLE001
        print(f"Error in retrieve_and_generate API: {e}")


def display_rag_results_with_metadata(
    response: dict, query: str, metadata_filtering_used: bool = False
) -> None:
    """
    Pretty-print the generated answer, citations, and metadata.

    :param response: The raw response dict from retrieve_and_generate
    :param query: The original natural-language query
    :param metadata_filtering_used: Whether we enabled implicit metadata filtering
    """
    print(f"\nQUERY: {query}")
    print("-" * 40)

    if "output" not in response:
        print("No response generated")
        return

    # Print the model's natural-language answer
    print("GENERATED RESPONSE:")
    print("-" * 40)
    print(response["output"]["text"])

    # Show how many citations came back
    if "citations" not in response:
        print("\nNo citations returned.")
        return

    citations = response["citations"]
    print(f"\nCITATIONS ({len(citations)} found):")
    print("-" * 40)

    for i, citation in enumerate(citations, start=1):
        print(f"\nCitation {i}:")

        if "retrievedReferences" not in citation:
            continue

        refs = citation["retrievedReferences"]
        for j, ref in enumerate(refs, start=1):
            print(f"  Reference {j}:")

            # Show a small preview of the matched text
            content = ref.get("content", {})
            if "text" in content:
                text = content["text"]
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"    Content: {preview}")

            # Show any attached metadata
            if "metadata" in ref:
                metadata = ref["metadata"]
                print("    Metadata:")
                for key, value in metadata.items():
                    print(f"      {key}: {value}")

            # Show where in S3 the chunk came from, if present
            if "location" in ref and "s3Location" in ref["location"]:
                s3_info = ref["location"]["s3Location"]
                print(f"    Source: {s3_info.get('uri', 'N/A')}")

            # Show the retrieval score if provided
            if "score" in ref:
                print(f"    Relevance Score: {ref['score']}")


# -------------------------------------------------------------------------
# Example test queries (same spirit as the notebook cells)
# -------------------------------------------------------------------------


def run_example_queries():
    """Execute a few representative test questions against the KB."""
    # No metadata filtering – general view on a specific product
    run_unstructured_kb_query(
        "What specific problems or benefits do customers describe when reviewing product_890?"
    )

    # With metadata filtering – same question, but using metadata to narrow focus
    run_unstructured_kb_query(
        "What specific problems or benefits do customers describe when reviewing product_890?",
        use_metadata_filtering=True,
    )

    # Focus on cookbook-style products (higher-level semantic slice)
    run_unstructured_kb_query(
        "What do customers think about cookbook quality and recipe clarity?",
        use_metadata_filtering=True,
    )

    # Focus on highly-rated furniture
    run_unstructured_kb_query(
        "What features make customers give 4- and 5-star ratings to furniture products?",
        use_metadata_filtering=True,
    )


if __name__ == "__main__":
    run_example_queries()
