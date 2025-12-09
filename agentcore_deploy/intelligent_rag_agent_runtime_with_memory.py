# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import boto3
import logging
from strands import Agent, tool
from strands.telemetry import StrandsTelemetry
from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

# Initialize OpenTelemetry for observability
# AgentCore sets OTEL_EXPORTER_OTLP_ENDPOINT, otherwise default to AWS X-Ray
if not os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT'):
    region = boto3.Session().region_name
    os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = f'https://xray.{region}.amazonaws.com/v1/traces'

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
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

app = BedrockAgentCoreApp()
modelID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# Get configuration from SSM
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
        return {
            'unstructured_kb_id': os.environ.get('UNSTRUCTURED_KB_ID'),
            'structured_kb_id': os.environ.get('STRUCTURED_KB_ID')
        }

def get_memory_id():
    try:
        ssm = boto3.client("ssm")
        memory_id = ssm.get_parameter(Name="/app/intelligent_rag/agentcore/memory_id")
        return memory_id["Parameter"]["Value"]
    except Exception as e:
        print(f"Error retrieving Memory ID from SSM: {e}")
        return os.environ.get('MEMORY_ID')

# Initialize AWS clients
region = boto3.Session().region_name
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=region)
sts_client = boto3.client('sts')
account_id = sts_client.get_caller_identity()["Account"]

# Get configuration
kb_config = get_kb_config()
UNSTRUCTURED_KB_ID = kb_config['unstructured_kb_id']
STRUCTURED_KB_ID = kb_config['structured_kb_id']
MEMORY_ID = get_memory_id()

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
        
        # to understand `$output_format_instructions$` and other Bedrock KnowledgeBase macros
        # please refer to `Knowledge base prompt templates: orchestration & generation` at
        # https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-config.html?utm_source=chatgpt.com
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

@app.entrypoint
def invoke(payload, context):
    """Process user input with AgentCore Memory integration"""
    
    user_message = payload.get("prompt", "How many customers reviewed product_890?")
    user_id = payload.get("user_id", "default-user")
    session_id = context.session_id
    
    if not session_id:
        raise Exception("Context session_id is not set")
    
    logger.info(f"Intelligent RAG runtime - user_message: {user_message}")
    logger.info(f"Intelligent RAG runtime - session_id: {session_id}")
    logger.info(f"Intelligent RAG runtime - user_id: {user_id}")
    
    # Configure AgentCore Memory with retrieval settings
    memory_config = AgentCoreMemoryConfig(
        memory_id=MEMORY_ID,
        session_id=session_id,
        actor_id=user_id,
        retrieval_config={
            "/summaries/{actorId}/{sessionId}": RetrievalConfig(
                top_k=5,
                relevance_score=0.5
            ),
            "/users/{actorId}/preferences": RetrievalConfig(
                top_k=5,
                relevance_score=0.7
            )
        }
    )
    
    # Create session manager for automatic memory persistence
    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=memory_config,
        region_name=region
    )
    
    # Create agent with session manager - conversations automatically persisted!
    agent = Agent(
        system_prompt="""You are an intelligent assistant that routes queries to the appropriate knowledge base. 
        Choose the appropriate tool based on the query type. The tools return raw data that you should analyze 
        and present in a clear, helpful format. Use your memory to provide personalized responses based on 
        previous interactions.""",
        tools=[unstructured_data_assistant, structured_data_assistant],
        model=modelID,
        session_manager=session_manager
    )
    
    result = agent(user_message)
    return {"result": result.message}

if __name__ == "__main__":
    app.run()
