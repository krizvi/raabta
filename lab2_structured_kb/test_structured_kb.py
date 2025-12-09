#!/usr/bin/env python
# coding: utf-8

"""
# Test Structured Amazon Bedrock Knowledge Base

This notebook demonstrates testing the structured Amazon Bedrock...d in the previous notebook using the `RetrieveAndGenerate` API.

## Data Schema Overview

Our e-commerce database contains the following tables:
- **orders**: order_id, customer_id, order_total, order_status, payment_method, shipping_address, created_at, updated_at
- **order_items**: order_item_id, order_id, product_id, quantity, price
- **reviews**: review_id, product_id, customer_id, rating, created_at
- **payments**: payment_id, order_id, customer_id, amount, payment_method, payment_status, created_at
"""

# ------------------------------------------------------------------------------
# ## Setup and Prerequisites
#
# ### Prerequisites
# * Completed `2.1-prerequisites-structured-kb.ipynb` notebook
# * Structured Knowledge Base successfully created and synced
#
# Let's start by importing the required libraries and loading the Knowledge Base configuration:

# ------------------------------------------------------------------------------
# Import required libraries for AWS service interaction and testing the structured Knowledge Base query capabilities.

import json
import logging
import os
import time
from datetime import datetime

import boto3

session = boto3.session.Session()
region = session.region_name
sts_client = boto3.client("sts")
account_id = sts_client.get_caller_identity()["Account"]
# Initialize the bedrock agent runtime client
bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime")

print(f"AWS Region: {region}")
print(f"AWS Account ID: {account_id}")

ssm_client = boto3.client("ssm", region_name=region)


def load_structured_kb_config():
    """Resolve the structured KB ID from environment or SSM and print it.

    Order of precedence:
    1. STRUCTURED_KB_ID environment variable
    2. SSM parameter /app/intelligent_rag/agentcore/structured_kb_id
    """
    kb_id = os.getenv("STRUCTURED_KB_ID")
    if not kb_id:
        try:
            resp = ssm_client.get_parameter(
                Name="/app/intelligent_rag/agentcore/structured_kb_id"
            )
            kb_id = resp["Parameter"]["Value"]
        except ssm_client.exceptions.ParameterNotFound:
            raise RuntimeError(
                "Structured KB ID not found. "
                "Either set STRUCTURED_KB_ID or run the prerequisites script first."
            )

    print("Structured Knowledge Base Configuration:")
    print("=" * 50)
    print(f"KB ID: {kb_id}")
    print(f"Region: {region}")
    print("=" * 50)
    return kb_id


structured_kb_id = load_structured_kb_config()

# ## Model Configuration
#
# We will use the same foundation model that was configured for the structured Knowledge Base.

foundation_model = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

print(f"Using foundation model: {foundation_model}")

# ## Helper to Display RAG Results
#
# The result from retrieve_and_generate includes generated text and citations.
# For structured KB, citations often contain SQL queries that were executed against Redshift.
# We display those queries, but avoid printing duplicates.


def display_rag_results(response, query):
    """Display results from the retrieve_and_generate API, avoiding duplicate SQL queries."""
    print(f"\nQUERY: {query}")
    print("-" * 40)

    if "output" in response:
        print("GENERATED RESPONSE:")
        print("-" * 40)
        print(response["output"]["text"])

        if "citations" in response:
            citations = response["citations"]

            # Track seen SQL queries to avoid repeats
            seen_sql_queries = set()

            for i, citation in enumerate(citations, 1):
                print(f"\nCitation {i}:")

                references = []
                if "retrievedReferences" in citation:
                    references = citation["retrievedReferences"]
                elif "content" in citation and "location" in citation:
                    # Wrap top-level citation as a single reference for uniformity
                    references = [citation]
                else:
                    references = []
                for j, ref in enumerate(references, 1):
                    # Try to print SQL query if present
                    sql_query = None
                    # Check for SQL location in reference
                    location = ref.get("location", {})
                    if "sqlLocation" in location:
                        sql_query = location["sqlLocation"].get("query", None)
                    # Sometimes, for top-level citation, location is at citation level
                    elif "sqlLocation" in citation.get("location", {}):
                        sql_query = citation["location"]["sqlLocation"].get(
                            "query", None
                        )
                    if sql_query:
                        if sql_query not in seen_sql_queries:
                            print(f"    SQL Query: {sql_query}")
                            seen_sql_queries.add(sql_query)
                        # else: do not print anything for duplicates
                    else:
                        print(f"    SQL Query: (not found)")
    else:
        print("No response generated")


def run_structured_kb_query(query):
    """Test a query using only the retrieve_and_generate API"""

    try:
        rag_response = bedrock_agent_runtime_client.retrieve_and_generate(
            input={"text": query},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": structured_kb_id,
                    "modelArn": f"arn:aws:bedrock:{region}:{account_id}:inference-profile/{foundation_model}",
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {"numberOfResults": 5}
                    },
                },
            },
        )
        display_rag_results(rag_response, query)
    except Exception as e:
        print(f"Error in retrieve_and_generate API: {str(e)}")


if __name__ == "__main__":
    # Query 1 – basic sanity check on total number of reviews
    run_structured_kb_query("How many reviews do we have in total?")
    # Query 2 – check 1-star ratings per product
    run_structured_kb_query(
        "Which product_ids received the most 1-star ratings in 2022?"
    )
