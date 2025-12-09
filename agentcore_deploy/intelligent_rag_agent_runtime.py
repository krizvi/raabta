# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import boto3
import logging
from strands import Agent, tool
from strands.telemetry import StrandsTelemetry

# AgentCore imports
from bedrock_agentcore import BedrockAgentCoreApp

# Initialize OpenTelemetry for observability
# AgentCore sets OTEL_EXPORTER_OTLP_ENDPOINT, otherwise default to AWS X-Ray
if not os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT'):
    region = boto3.Session().region_name
    os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = f'https://xray.{region}.amazonaws.com/v1/traces'

strands_telemetry = StrandsTelemetry()
strands_telemetry.setup_otlp_exporter()

# Initialize OpenTelemetry for observability
strands_telemetry = StrandsTelemetry()
strands_telemetry.setup_otlp_exporter()

# Set up logging for Strands components
loggers = [
    'strands',
    'strands.agent', 
    'strands.tools', 
    'strands.models', 
    'strands.bedrock'
    ]
for logger_name in loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # Add console handler if not already present
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

app = BedrockAgentCoreApp()

modelID="global.anthropic.claude-haiku-4-5-20251001-v1:0"


# Get configuration from environment or SSM
def get_kb_config():
    try:
        ssm = boto3.client("ssm")
        unstructured_kb = ssm.get_parameter(Name="/app/intelligent_rag/agentcore/unstructured_kb_id")
        structured_kb = ssm.get_parameter(Name="/app/intelligent_rag/agentcore/structured_kb_id")
        
        return {
            'unstructured_kb_id': unstructured_kb["Parameter"]["Value"],
            'structured_kb_id': structured_kb["Parameter"]["Value"]
        }
    except Exception as e:
        print(f"Error retrieving KB IDs from SSM: {e}")
        # Fallback to environment variables
        return {
            'unstructured_kb_id': os.environ.get('UNSTRUCTURED_KB_ID'),
            'structured_kb_id': os.environ.get('STRUCTURED_KB_ID')
        }

# Initialize AWS clients
region = boto3.Session().region_name
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=region)
sts_client = boto3.client('sts')
account_id = sts_client.get_caller_identity()["Account"]
boto3.setup_default_session()

# Get KB configuration
kb_config = get_kb_config()
UNSTRUCTURED_KB_ID = kb_config['unstructured_kb_id']
STRUCTURED_KB_ID = kb_config['structured_kb_id']

@tool
def unstructured_data_assistant(query: str) -> str:
    """
    Handle document-based, narrative, and conceptual queries using the unstructured knowledge base.
    
    Use this tool for:
    - Customer review analysis and sentiment
    - Product feedback and quality insights
    - Qualitative questions about user experiences
    - Document comprehension and content analysis
    
    Args:
        query: A question about customer reviews, product feedback, user experiences,
               or requiring document comprehension and qualitative analysis
    
    Returns:
        Retrieved context from unstructured knowledge base for agent processing
    """
    try:
        # Configure retrieval with metadata filtering
        retrieval_config = {
            "vectorSearchConfiguration": {
                "numberOfResults": 10,
                "implicitFilterConfiguration": {
                    "metadataAttributes": [
                        {
                            "key": "product_type",
                            "type": "STRING",
                            "description": "The type of product being reviewed. Possible values include: 'cookbook', 'kitchenware', 'furniture', 'speaker', 'educational toy', 'board game', 'shirt', 'self-help'"
                        },
                        {
                            "key": "rating",
                            "type": "NUMBER",
                            "description": "The rating given by the customer, ranging from 1 to 5 stars"
                        },
                        {
                            "key": "created_at",
                            "type": "STRING",
                            "description": "The date when the review was created in YYYY-MM-DD format"
                        },
                        {
                            "key": "product_id",
                            "type": "STRING",
                            "description": "The unique identifier of the product being reviewed"
                        },
                        {
                            "key": "customer_id",
                            "type": "STRING",
                            "description": "The unique identifier of the customer who wrote the review"
                        }
                    ],
                    "modelArn": f"arn:aws:bedrock:{region}:{account_id}:inference-profile/{modelID}"
                }
            }
        }
        
        # Use retrieve API - let the agent handle response generation
        retrieved_response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=UNSTRUCTURED_KB_ID,
            retrievalQuery={'text': query + "\n$output_format_instructions$"},
            retrievalConfiguration=retrieval_config
        )
        
        return retrieved_response
        
    except Exception as e:
        return f"Error in unstructured data assistant: {str(e)}"

@tool
def structured_data_assistant(query: str) -> str:
    """
    Handle data analysis, metrics, and quantitative queries using the structured knowledge base.
    
    Args:
        query: A question requiring calculations, aggregations, statistical analysis,
               or database operations on structured data
    
    Returns:
        Raw retrieve response from the structured knowledge base
    """
    try:
        retrieve_response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=STRUCTURED_KB_ID,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': 10,
                }
            }
        )
        
        return retrieve_response
        
    except Exception as e:
        return f"Error in structured data assistant: {str(e)}"




def create_intelligent_rag_agent():
    """Create the intelligent RAG agent with routing capabilities"""
    
    system_prompt="""You are an intelligent assistant that routes queries to the appropriate knowledge base. Choose the appropriate tool based on the query type. 
    The tools return raw data that you should analyze and present in a clear, helpful format."""
    
    agent = Agent(
        system_prompt=system_prompt,
        tools=[unstructured_data_assistant, structured_data_assistant],
        model=modelID
    )
    
    return agent

# Create the agent instance
intelligent_rag_agent = create_intelligent_rag_agent()

@app.entrypoint
def invoke(payload):
    """Process user input and return a response"""
    user_message = payload.get("prompt", "How many customers reviewed product_890, are those reviews positive or negative?")
    result = intelligent_rag_agent(user_message)
    return {"result": result.message}

if __name__ == "__main__":
    app.run()
