#!/usr/bin/env python
# coding: utf-8

"""
# Create Structured Amazon Bedrock Knowledge Base with Redshift

This notebook demonstrates how to create and configure an Amazon...ses Amazon Redshift Serverless as a source for structured data.

The Knowledge Base integrates Amazon Redshift as the data source...ata including orders, payments, reviews, and customer analytics.

This structured knowledge base will be used in conjunction with ...ctured knowledge base to create agentic RAG using Strands Agents
"""


# ![Structured Knowledge Base](../images/structured_kb.png)

# ## Setup and Prerequisites
#
# ### Prerequisites
# * Python 3.13
# * AWS account with appropriate permissions
# * Amazon Bedrock foundation models enabled
# * IAM permissions for Amazon Redshift Serverless, Amazon S3, and Amazon Bedrock
#
# ### Required AWS Services
# - **Amazon Bedrock**: For knowledge base creation and LLM inference
# - **Amazon Redshift Serverless**: As the structured data source
# - **Amazon S3**: For data staging and intermediate storage
# - **AWS IAM**: For service permissions and roles
#
# Let's start by importing the required libraries and setting up AWS clients:

# Import required libraries for AWS service interaction, Redshift management, and data handling:

import json
import logging
import os
import random
import string
import time
import uuid
from datetime import datetime

import boto3
import requests

# Initialize AWS service clients for S3, STS, IAM, Redshift Serverless, and Bedrock:

session = boto3.session.Session()
region = session.region_name
if region is None:
    raise RuntimeError(
        "No AWS region set. Configure AWS_REGION or default region first."
    )

s3_client = boto3.client("s3", region_name=region)
sts_client = boto3.client("sts", region_name=region)
iam_client = boto3.client("iam", region_name=region)
redshift_client = boto3.client("redshift-serverless", region_name=region)
bedrock_client = boto3.client("bedrock", region_name=region)

account_id = sts_client.get_caller_identity()["Account"]
print(f"AWS Region: {region}")
print(f"AWS Account ID: {account_id}")

# Configure logging for detailed monitoring of resource creation and operations:

logging.basicConfig(
    format="[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ## Step 0: Generate unique suffix for resource names
#
# To avoid naming conflicts when multiple participants run the workshop simultaneously,
# we generate a unique random suffix for AWS resource names.

# Generate unique suffix for resource names
suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

print(f"Using suffix: {suffix}")

# ## Step 1: Import Amazon Bedrock Knowledge Bases helper
#
# Lets import the structured knowledge base utility to help with Knowledge Base configuration and creation.

import os

if "Lab 2" in os.getcwd():
    os.chdir("..")
else:
    print(os.getcwd())

from utils.structured_knowledge_base import BedrockStructuredKnowledgeBase

# ## Step 2: Set up Redshift Serverless Infrastructure
#
# Next we will create the necessary Redshift Serverless components needed to host our structured data.
# We will create:
# - A **namespace**: logical grouping of database objects and users
# - A **workgroup**: compute resources for queries
#
# We will also create an S3 bucket for data staging, and IAM roles for Redshift to access S3.

# Configuration for Redshift resources
REDSHIFT_NAMESPACE = f"sds-ecommerce-{suffix}"
REDSHIFT_WORKGROUP = f"sds-ecommerce-wg-{suffix}"
REDSHIFT_DATABASE = f"sds-ecommerce"
S3_BUCKET = f"sds-ecommerce-redshift-{suffix}"

print(f"Redshift Namespace: {REDSHIFT_NAMESPACE}")
print(f"Redshift Workgroup: {REDSHIFT_WORKGROUP}")
print(f"Database: {REDSHIFT_DATABASE}")
print(f"S3 Bucket: {S3_BUCKET}")

# ## Create IAM Role for Redshift
#
# Redshift needs an IAM role to read data from S3 when we run COPY commands.
# The following function creates (or reuses) that IAM role and attaches AmazonS3ReadOnlyAccess.


def create_iam_role_for_redshift():
    """Create IAM role for Redshift to access S3"""
    try:
        # Get account ID
        account_id = sts_client.get_caller_identity()["Account"]

        # Create IAM role if it doesn't exist
        role_name = f"RedshiftS3AccessRole-{suffix}"
        try:
            role_response = iam_client.get_role(RoleName=role_name)
            print(f"Role {role_name} already exists")
            return f"arn:aws:iam::{account_id}:role/{role_name}"
        except iam_client.exceptions.NoSuchEntityException:
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "redshift-serverless.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }

            iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="Role for Redshift to access S3",
            )

            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",
            )

            print(f"Created role {role_name}")
            return f"arn:aws:iam::{account_id}:role/{role_name}"

    except Exception as e:
        print(f"Error creating IAM role: {str(e)}")
        raise


redshift_role_arn = create_iam_role_for_redshift()
print(f"Redshift IAM Role ARN: {redshift_role_arn}")

# ## Create Redshift Serverless Namespace and Workgroup
#
# Now we create the Redshift Serverless namespace and workgroup required for hosting the database.


def create_redshift_namespace(namespace_name, admin_user, admin_password, iam_role_arn):
    """Create Redshift Serverless namespace with the given admin credentials"""
    try:
        # Check if namespace already exists
        namespaces = redshift_client.list_namespaces()["namespaces"]
        for ns in namespaces:
            if ns["namespaceName"] == namespace_name:
                print(f"Namespace {namespace_name} already exists")
                return ns["namespaceArn"]

        # Create new namespace
        response = redshift_client.create_namespace(
            namespaceName=namespace_name,
            adminUsername=admin_user,
            adminUserPassword=admin_password,
            iamRoles=[iam_role_arn],
        )
        print(f"Creating namespace {namespace_name}...")
        return response["namespace"]["namespaceArn"]
    except Exception as e:
        print(f"Error creating namespace: {str(e)}")
        raise


def wait_for_namespace_available(namespace_name):
    """Wait for Redshift namespace to become available"""
    print(f"Waiting for namespace {namespace_name} to become available...")
    while True:
        response = redshift_client.get_namespace(namespaceName=namespace_name)
        status = response["namespace"]["status"]
        print(f"  Namespace status: {status}")
        if status == "AVAILABLE":
            print("Namespace is now available")
            return response["namespace"]["namespaceArn"]
        elif status == "DELETING":
            raise RuntimeError(
                "Namespace is being deleted. Please use a different name."
            )
        time.sleep(30)


def create_redshift_workgroup(workgroup_name, namespace_name, base_capacity=32):
    """Create Redshift Serverless workgroup"""
    try:
        # Check if workgroup already exists
        workgroups = redshift_client.list_workgroups()["workgroups"]
        for wg in workgroups:
            if wg["workgroupName"] == workgroup_name:
                print(f"Workgroup {workgroup_name} already exists")
                return wg["workgroupArn"]

        response = redshift_client.create_workgroup(
            workgroupName=workgroup_name,
            namespaceName=namespace_name,
            baseCapacity=base_capacity,
        )
        print(f"Creating workgroup {workgroup_name}...")
        return response["workgroup"]["workgroupArn"]
    except Exception as e:
        print(f"Error creating workgroup: {str(e)}")
        raise


def wait_for_workgroup_available(workgroup_name):
    """Wait for Redshift workgroup to become available"""
    print(f"Waiting for workgroup {workgroup_name} to become available...")
    while True:
        response = redshift_client.get_workgroup(workgroupName=workgroup_name)
        status = response["workgroup"]["status"]
        print(f"  Workgroup status: {status}")
        if status == "AVAILABLE":
            print("Workgroup is now available")
            return response["workgroup"]["workgroupArn"]
        elif status == "DELETING":
            raise RuntimeError(
                "Workgroup is being deleted. Please use a different name."
            )
        time.sleep(30)


# Prompt for admin credentials for the Redshift namespace
# In a production or automated environment, you would not hardcode or prompt,
# but instead provide from a secure source. Here it was likely provided inline
# in the notebook or via parameters.

admin_user = "admin"
admin_password = "Admin123456!"  # NOTE: demo only; use a secret manager in real life

namespace_arn = create_redshift_namespace(
    REDSHIFT_NAMESPACE, admin_user, admin_password, redshift_role_arn
)

namespace_arn = wait_for_namespace_available(REDSHIFT_NAMESPACE)
print(f"Namespace ARN: {namespace_arn}")

workgroup_arn = create_redshift_workgroup(REDSHIFT_WORKGROUP, REDSHIFT_NAMESPACE)
workgroup_arn = wait_for_workgroup_available(REDSHIFT_WORKGROUP)
print(f"Workgroup ARN: {workgroup_arn}")

# ## Step 3: Create S3 Bucket for Structured Data
#
# We'll create an S3 bucket to stage our structured CSV data before loading into Redshift.
# The bucket name is unique per workshop using the generated suffix.


def create_s3_bucket(bucket_name, region_name):
    """Create an S3 bucket in the specified region if it does not already exist."""
    try:
        if region_name == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region_name},
            )
        print(f"Bucket {bucket_name} created successfully")
    except s3_client.exceptions.BucketAlreadyOwnedByYou:
        print(f"Bucket {bucket_name} already exists and is owned by you")
    except Exception as e:
        print(f"Error creating S3 bucket: {str(e)}")
        raise


create_s3_bucket(S3_BUCKET, region)

# ## Step 4: Upload Structured Sample Data to S3
#
# This step uploads local CSV files representing orders, order items, reviews, and payments to S3.
# Redshift will load these CSVs via COPY commands.


def upload_csv_to_s3(local_path, bucket_name, key):
    """Upload a local CSV file to S3 under the given key."""
    print(f"Uploading {local_path} to s3://{bucket_name}/{key}")
    s3_client.upload_file(local_path, bucket_name, key)
    print(f"Uploaded {local_path}")


# Assuming the CSV files are under a local data directory in the repo:
data_base_dir = "sample_structured_data"
csv_files = {
    "orders.csv": "orders/orders.csv",
    "order_items.csv": "orders/order_items.csv",
    "reviews.csv": "orders/reviews.csv",
    "payments.csv": "orders/payments.csv",
}

for local_name, s3_key in csv_files.items():
    local_path = os.path.join(data_base_dir, local_name)
    upload_csv_to_s3(local_path, S3_BUCKET, s3_key)

# ## Step 5: Connect to Redshift and Create Schema + Tables
#
# We will connect to the Redshift Serverless endpoint using the workgroup, and
# create the schema and tables required for the e-commerce dataset.

import psycopg2


def get_redshift_connection(database, user, password, workgroup_name, region_name):
    """Get a psycopg2 connection to Redshift Serverless workgroup."""
    # Get workgroup details to resolve endpoint
    wg = redshift_client.get_workgroup(workgroupName=workgroup_name)["workgroup"]
    host = wg["endpoint"]["address"]
    port = wg["endpoint"]["port"]
    print(f"Connecting to Redshift at {host}:{port}, database {database}")
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=database,
        user=user,
        password=password,
        sslmode="require",
    )
    return conn


def create_database_if_not_exists(admin_conn, database_name):
    """Create the workshop database if it does not exist."""
    admin_conn.autocommit = True
    cur = admin_conn.cursor()
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{database_name}'")
    exists = cur.fetchone() is not None
    if exists:
        print(f"Database {database_name} already exists.")
    else:
        print(f"Creating database {database_name}...")
        cur.execute(f"CREATE DATABASE {database_name}")
    cur.close()


# Connect to the default 'dev' database as admin, create our workshop database if needed
admin_conn = get_redshift_connection(
    "dev", admin_user, admin_password, REDSHIFT_WORKGROUP, region
)
create_database_if_not_exists(admin_conn, REDSHIFT_DATABASE)
admin_conn.close()

# Now connect directly to our workshop database
conn = get_redshift_connection(
    REDSHIFT_DATABASE, admin_user, admin_password, REDSHIFT_WORKGROUP, region
)
cursor = conn.cursor()


# Create tables in Redshift
def create_tables():
    """Create all necessary tables in Redshift"""

    # Orders table
    orders_sql = """
    CREATE TABLE IF NOT EXISTS orders (
        order_id VARCHAR(255) PRIMARY KEY,
        customer_id VARCHAR(255),
        order_total DECIMAL(10,2),
        order_status VARCHAR(50),
        payment_method VARCHAR(50),
        shipping_address TEXT,
        created_at TIMESTAMP,
        updated_at TIMESTAMP
    )
    """

    # Order items table
    order_items_sql = """
    CREATE TABLE IF NOT EXISTS order_items (
        order_item_id VARCHAR(255) PRIMARY KEY,
        order_id VARCHAR(255),
        product_id VARCHAR(255),
        quantity INT,
        price DECIMAL(10,2)
    )
    """

    # Reviews table
    reviews_sql = """
    CREATE TABLE IF NOT EXISTS reviews (
        review_id VARCHAR(255) PRIMARY KEY,
        product_id VARCHAR(255),
        customer_id VARCHAR(255),
        rating INT,
        created_at TIMESTAMP
    )
    """

    # Payments table
    payments_sql = """
    CREATE TABLE IF NOT EXISTS payments (
        payment_id VARCHAR(255) PRIMARY KEY,
        order_id VARCHAR(255),
        customer_id VARCHAR(255),
        amount DECIMAL(10,2),
        payment_method VARCHAR(50),
        payment_status VARCHAR(50),
        created_at TIMESTAMP
    )
    """

    print("Creating tables in Redshift...")
    cursor.execute(orders_sql)
    cursor.execute(order_items_sql)
    cursor.execute(reviews_sql)
    cursor.execute(payments_sql)
    conn.commit()
    print("Tables created successfully!")


create_tables()

# ## Load Data into Redshift from S3
#
# We now populate the tables by running COPY commands that pull CSV data from S3.


def copy_table_from_s3(table_name, s3_key, iam_role_arn, bucket_name):
    """COPY data from S3 into a Redshift table"""
    copy_sql = f"""
    COPY {table_name}
    FROM 's3://{bucket_name}/{s3_key}'
    IAM_ROLE '{iam_role_arn}'
    CSV
    IGNOREHEADER 1
    REGION '{region}'
    TIMEFORMAT 'auto';
    """
    print(f"Running COPY for table {table_name} from s3://{bucket_name}/{s3_key}")
    cursor.execute(copy_sql)
    conn.commit()
    print(f"Data loaded into {table_name}")


copy_table_from_s3("orders", "orders/orders.csv", redshift_role_arn, S3_BUCKET)
copy_table_from_s3(
    "order_items", "orders/order_items.csv", redshift_role_arn, S3_BUCKET
)
copy_table_from_s3("reviews", "orders/reviews.csv", redshift_role_arn, S3_BUCKET)
copy_table_from_s3("payments", "orders/payments.csv", redshift_role_arn, S3_BUCKET)

# ## Step 6: Configure Structured Knowledge Base
#
# With Redshift populated, we now configure the Amazon Bedrock Structured Knowledge Base
# to point at this Redshift dataset and enable structured RAG queries.

# Foundation and embedding models for structured KB
foundation_model = "anthropic.claude-haiku-4-5-20251001-v1:0"
generation_model = "global." + foundation_model

# Define the knowledge base configuration parameters specific to Redshift:
kb_config_param = {
    "redshift": {
        "cluster_id": None,  # not used for Serverless
        "workgroup_arn": None,  # will be set at instantiation
        "database_name": REDSHIFT_DATABASE,
        "db_user": admin_user,
    }
}

try:
    # Create structured knowledge base instance
    structured_kb = BedrockStructuredKnowledgeBase(
        kb_name=f"sds-structured-kb-{suffix}",
        kb_description="E-commerce structured data knowledge base using Redshift Serverless",
        redshift_database=REDSHIFT_DATABASE,
        redshift_namespace=REDSHIFT_NAMESPACE,
        redshift_workgroup=REDSHIFT_WORKGROUP,
        redshift_role_arn=redshift_role_arn,
        workgroup_arn=workgroup_arn,
        kbConfigParam=kb_config_param,
        generation_model=generation_model,
        suffix=suffix,
    )

    print("Knowledge Base created successfully!")
    kb_id = structured_kb.get_knowledge_base_id()
    print(f"Knowledge Base ID: {kb_id}")

except Exception as e:
    print(f"Error creating Knowledge Base: {str(e)}")
    raise

# ## Store Structured Knowledge Base Configuration
#
# After successful creation, we store the structured KB ID and related configuration
# so it can be reused in the test notebook. In the original notebook this was stored
# in Jupyter using %store; here we keep variables in the script and optionally in SSM.

# Store the structured knowledge base configuration
structured_kb_id = structured_kb.get_knowledge_base_id()
structured_kb_region = region
structured_workgroup_arn = workgroup_arn
structured_database_name = REDSHIFT_DATABASE

# Store variables for use in main notebook

print("=" * 60)
print(f"Structured Knowledge Base ID: {structured_kb_id}")
print(f"Region: {structured_kb_region}")
print(f"Workgroup ARN: {structured_workgroup_arn}")
print(f"Database Name: {structured_database_name}")
print("=" * 60)
print("Configuration stored successfully!")

# Display the Knowledge Base ID to manually copy if needed

print("\nStructured KB setup complete. You can now use this KB ID for testing:")
print(f"KB ID: {structured_kb_id}")

# Finally, store the KB ID in SSM Parameter Store so that other scripts can
# discover it without relying on notebook %store variables.

param_name = "/app/intelligent_rag/agentcore/structured_kb_id"

ssm = boto3.client("ssm")
ssm.put_parameter(
    Name=param_name, Value=structured_kb_id, Type="String", Overwrite=True
)
print(f"Stored {structured_kb_id} in SSM: {param_name}")

# ## Summary
#
# If all the above executed successfully, you have:
#
# - Created Amazon Redshift Serverless namespace and workgroup infrastructure
# - Set up an S3 bucket and uploaded sample structured data
# - Created database tables and loaded data from S3 using COPY commands
# - Created an Amazon Bedrock Knowledge Base configured for structured data queries
# - Stored the KB ID in SSM for later testing
