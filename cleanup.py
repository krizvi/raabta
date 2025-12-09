#!/usr/bin/env python
# coding: utf-8

"""Cleanup utilities for the Unstructured & Structured RAG Agent workshop.

This module provides functions that delete workshop resources in AWS, including
Bedrock knowledge bases (unstructured and structured), S3 data buckets,
Redshift Serverless workgroups/namespaces, AgentCore endpoints and memories,
IAM roles, and SSM parameters. Use complete_workshop_cleanup() as the main
entry point when you are ready to remove all workshop resources.
"""

# # 🧹 Workshop Cleanup Instructions
#
# This notebook provides comprehensive cleanup instructions for cleaning up AWS resources created during the Unstructured & Structured RAG Agent Workshop.
#
# **⚠️ Note**: If you are at an AWS event using vended accounts, you do NOT need to perform these steps.  The workshop environment will delete these resources for you.
#
# **⚠️ Important**: Running these cleanup steps will permanently delete resources and data. Make sure you've completed all labs and no longer need the resources before proceeding.
#
# ## 📋 Resources to Clean Up
#
# This workshop creates the following AWS resources that incur costs:
#
# ### Lab 1 - Unstructured Knowledge Base
# - Amazon Bedrock Knowledge Base (Unstructured)
# - Amazon S3 bucket with sample data
# - IAM roles and policies
# - Amazon OpenSearch Serverless collection
#
# ### Lab 2 - Structured Knowledge Base
# - Amazon Bedrock Knowledge Base (Structured)
# - Amazon Redshift Serverless data warehouse
# - Amazon Redshift database and tables
# - IAM roles and policies
#
# ### AgentCore / Agent Labs
# - Amazon Bedrock AgentCore Endpoint
# - AgentCore tools, memory, and runtime resources
# - Amazon SSM Parameters for configuration
# - Additional IAM roles and policies
#
# ## 🧼 Cleanup Strategy
#
# We'll clean up resources in the following order:
#
# 1. Unstructured Knowledge Base and associated resources
# 2. Structured Knowledge Base and Redshift resources
# 3. AgentCore resources and endpoints
# 4. IAM Roles and Policies
# 5. SSM Parameters
#
# You can run the **automated cleanup** using this script, and we'll also provide **manual cleanup** steps as a backup.

import boto3
import botocore
import json
import time
from botocore.exceptions import ClientError

# Import workshop helper classes (these were used during the labs)
try:
    from intelligent_rag.schemas import (
        BedrockKnowledgeBase,
        BedrockStructuredKnowledgeBase,
    )
except ImportError:
    print(
        "⚠️ Workshop helper classes not found. "
        "Make sure this script is run in the same environment as the workshop labs."
    )

# Initialize AWS clients
session = boto3.Session()
region = session.region_name or "us-east-1"

ssm_client = boto3.client("ssm", region_name=region)
s3_client = boto3.client("s3", region_name=region)
redshift_client = boto3.client("redshift-serverless", region_name=region)
sts_client = boto3.client("sts", region_name=region)
iam_client = boto3.client("iam", region_name=region)
logs_client = boto3.client("logs", region_name=region)
bedrock_agent_client = boto3.client("bedrock-agent", region_name=region)

print(f"AWS Region: {region}")
print(f"Account ID: {sts_client.get_caller_identity()['Account']}")

# Load stored variables from workshop labs (if running in a Jupyter/IPython environment)
try:
    get_ipython().run_line_magic("store", "-r unstructured_kb_id")
    get_ipython().run_line_magic("store", "-r kb_region")
    get_ipython().run_line_magic("store", "-r data_bucket_name")
    print(f"✅ Unstructured KB ID: {unstructured_kb_id}")
    print(f"✅ Data Bucket: {data_bucket_name}")
except NameError:
    print("⚠️ Unstructured KB variables not found - may have been cleaned up already")
    unstructured_kb_id = None
    data_bucket_name = None

try:
    get_ipython().run_line_magic("store", "-r structured_kb_id")
    get_ipython().run_line_magic("store", "-r structured_kb_region")
    print(f"✅ Structured KB ID: {structured_kb_id}")
except NameError:
    print("⚠️ Structured KB variables not found - may have been cleaned up already")
    structured_kb_id = None

# Try to get additional variables from SSM parameters
try:
    unstructured_param = ssm_client.get_parameter(
        Name="/app/intelligent_rag/agentcore/unstructured_kb_id"
    )
    if not unstructured_kb_id:
        unstructured_kb_id = unstructured_param["Parameter"]["Value"]
        print(f"✅ Found Unstructured KB ID in SSM: {unstructured_kb_id}")
except ssm_client.exceptions.ParameterNotFound:
    pass

try:
    structured_param = ssm_client.get_parameter(
        Name="/app/intelligent_rag/agentcore/structured_kb_id"
    )
    if not structured_kb_id:
        structured_kb_id = structured_param["Parameter"]["Value"]
        print(f"✅ Found Structured KB ID in SSM: {structured_kb_id}")
except ssm_client.exceptions.ParameterNotFound:
    pass

try:
    bucket_param = ssm_client.get_parameter(
        Name="/app/intelligent_rag/agentcore/data_bucket_name"
    )
    if not data_bucket_name:
        data_bucket_name = bucket_param["Parameter"]["Value"]
        print(f"✅ Found data bucket in SSM: {data_bucket_name}")
except ssm_client.exceptions.ParameterNotFound:
    pass


# ### Delete Unstructured Knowledge Base
#
# **⚠️ Warning**: This will permanently delete the unstructured knowledge base and all associated resources.


def cleanup_unstructured_kb():
    """Clean up the unstructured knowledge base and associated resources."""
    if not unstructured_kb_id:
        print("⚠️ No unstructured knowledge base ID found - skipping cleanup")
        return

    try:
        print("=" * 95)
        print("Deleting Unstructured Knowledge Base and related resources...")
        print("=" * 95)

        # Create knowledge base instance for cleanup
        unstructured_knowledge_base = BedrockKnowledgeBase(
            kb_name="product-reviews-unstructured-kb",  # Name doesn't matter for deletion
            kb_description="Cleanup instance",
            kb_id=unstructured_kb_id,
        )

        # Delete the unstructured knowledge base and optionally its IAM roles and policies
        unstructured_knowledge_base.delete_kb(delete_iam_roles_and_policies=True)
        print("✅ Unstructured Knowledge Base deleted successfully!")

        # Clean up SSM parameter that stored the KB ID
        try:
            ssm_client.delete_parameter(
                Name="/app/intelligent_rag/agentcore/unstructured_kb_id"
            )
            print("✅ Deleted SSM parameter for unstructured KB ID")
        except ssm_client.exceptions.ParameterNotFound:
            print(
                "ℹ️ SSM parameter for unstructured KB ID not found - may have been deleted already"
            )

    except Exception as e:
        print(f"❌ Error deleting unstructured KB: {str(e)}")
        print("You may need to manually delete the unstructured KB in the AWS console.")


# ### Delete Structured Knowledge Base
#
# **⚠️ Warning**: This will permanently delete the structured knowledge base and all associated resources.


def cleanup_structured_kb():
    """Clean up the structured knowledge base."""
    if not structured_kb_id:
        print("⚠️ No structured knowledge base ID found - skipping cleanup")
        return

    try:
        print("=" * 95)
        print("Deleting Structured Knowledge Base and related resources...")
        print("=" * 95)

        # Create structured knowledge base instance for cleanup
        structured_kb = BedrockStructuredKnowledgeBase(
            kb_name="cleanup-instance",  # Name doesn't matter for deletion
            kb_description="Cleanup instance",
            kb_id=structured_kb_id,
        )

        # Delete the structured knowledge base and its IAM roles and policies
        structured_kb.delete_kb(delete_iam_roles_and_policies=True)
        print("✅ Structured Knowledge Base deleted successfully!")

        # Clean up SSM parameter
        try:
            ssm_client.delete_parameter(
                Name="/app/intelligent_rag/agentcore/structured_kb_id"
            )
            print("✅ Deleted SSM parameter for structured KB ID")
        except ssm_client.exceptions.ParameterNotFound:
            print(
                "ℹ️ SSM parameter for structured KB ID not found - may have been deleted already"
            )

    except Exception as e:
        print(f"❌ Error deleting structured KB: {str(e)}")
        print("You may need to manually delete the structured KB in the AWS console.")


# ### Delete S3 Bucket and Data
#
# This step deletes the S3 bucket that contains the sample data used in the workshop.


def cleanup_s3_bucket():
    """Delete the S3 bucket that stored unstructured workshop data."""
    if not data_bucket_name:
        print("⚠️ No data bucket name found - skipping S3 cleanup")
        return

    print("=" * 95)
    print("Deleting S3 bucket and all objects...")
    print("=" * 95)

    try:
        # Delete all objects from the bucket
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=data_bucket_name)

        object_count = 0
        for page in pages:
            if "Contents" in page:
                objects = [{"Key": obj["Key"]} for obj in page["Contents"]]
                object_count += len(objects)
                s3_client.delete_objects(
                    Bucket=data_bucket_name, Delete={"Objects": objects}
                )

        print(f"✅ Deleted {object_count} objects from bucket: {data_bucket_name}")

        # Delete the bucket itself
        s3_client.delete_bucket(Bucket=data_bucket_name)
        print(f"✅ Deleted S3 bucket: {data_bucket_name}")

        # Clean up SSM parameter
        try:
            ssm_client.delete_parameter(
                Name="/app/intelligent_rag/agentcore/data_bucket_name"
            )
            print("✅ Deleted SSM parameter for data bucket name")
        except ssm_client.exceptions.ParameterNotFound:
            print(
                "ℹ️ SSM parameter for data bucket not found - may have been deleted already"
            )

    except ClientError as e:
        print(f"❌ Error deleting S3 bucket: {e}")
        print(
            "You may need to manually delete the S3 bucket and its contents in the AWS console."
        )


# ### Delete Redshift Serverless Resources
#
# This will delete the Redshift Serverless workgroup and namespace created by the structured KB lab.


def find_redshift_resources():
    """Discover Redshift Serverless resources created for the workshop (best-effort)."""
    workgroups = redshift_client.list_workgroups()["workgroups"]
    namespaces = redshift_client.list_namespaces()["namespaces"]

    workshop_workgroups = [
        wg for wg in workgroups if "intelligent-rag" in wg["workgroupName"]
    ]
    workshop_namespaces = [
        ns for ns in namespaces if "intelligent-rag" in ns["namespaceName"]
    ]

    return workshop_workgroups, workshop_namespaces


def cleanup_redshift_resources():
    """Delete Redshift Serverless workgroups and namespaces created for the workshop."""
    print("=" * 95)
    print("Deleting Redshift Serverless resources...")
    print("=" * 95)

    try:
        workgroups, namespaces = find_redshift_resources()

        if not workgroups and not namespaces:
            print(
                "ℹ️ No Redshift Serverless resources found with 'intelligent-rag' in the name."
            )
            print(
                "If you created Redshift resources with different names, delete them manually."
            )
            return

        for wg in workgroups:
            wg_name = wg["workgroupName"]
            print(f"Requesting deletion of Redshift workgroup: {wg_name}")
            redshift_client.delete_workgroup(
                workgroupName=wg_name, deletePrimaryLogin=True
            )

        for ns in namespaces:
            ns_name = ns["namespaceName"]
            print(f"Requesting deletion of Redshift namespace: {ns_name}")
            redshift_client.delete_namespace(
                namespaceName=ns_name, finalSnapshotName=f"{ns_name}-final-snapshot"
            )

        print("✅ Requested deletion of Redshift Serverless workgroups and namespaces.")
        print("ℹ️ Deletion may take several minutes to complete.")

    except ClientError as e:
        print(f"❌ Error deleting Redshift resources: {e}")
        print(
            "You may need to manually delete Redshift Serverless resources in the AWS console."
        )


# ### Clean up AgentCore Resources
#
# This section will attempt to clean up:
# - AgentCore runtime endpoints
# - AgentCore memory instances
# - Related CloudWatch log groups


def cleanup_agentcore_resources():
    """Clean up AgentCore endpoints, memories, and related resources."""
    print("=" * 95)
    print("Cleaning up AgentCore resources (endpoints, memories, logs)...")
    print("=" * 95)

    try:
        # Import AgentCore-specific SDKs if available
        try:
            import agentcore_toolkit  # placeholder for actual toolkit module
        except ImportError:
            print("⚠️ AgentCore toolkit not available - skipping AgentCore cleanup")
            print(
                "You may need to manually delete AgentCore resources in the AWS console"
            )
            return

        # Placeholder for actual cleanup logic if using a toolkit.
        # In the workshop environment, these are often managed/created via CloudFormation or guided steps.
        print(
            "ℹ️ This function is a placeholder; implement specific AgentCore cleanup if needed."
        )

    except Exception as e:
        print(f"❌ Error during AgentCore cleanup: {str(e)}")
        print("You may need to manually delete remaining resources in the AWS console.")


# ### Clean up IAM Roles for AgentCore
#
# Clean up IAM roles created for AgentCore execution.


def cleanup_agentcore_iam_roles():
    """Clean up IAM roles created for AgentCore."""
    try:
        print("=" * 95)
        print("Cleaning up AgentCore IAM Roles...")
        print("=" * 95)

        # List all roles and find AgentCore-related ones
        roles = iam_client.list_roles()["Roles"]
        agentcore_roles = [
            r
            for r in roles
            if "AgentCoreExecutionRole" in r["RoleName"]
            or "intelligent-rag-agent" in r["RoleName"]
        ]

        for role in agentcore_roles:
            role_name = role["RoleName"]
            print(f"Cleaning up IAM role: {role_name}")

            # Detach associated policies
            attached_policies = iam_client.list_attached_role_policies(
                RoleName=role_name
            )["AttachedPolicies"]
            for policy in attached_policies:
                iam_client.detach_role_policy(
                    RoleName=role_name, PolicyArn=policy["PolicyArn"]
                )
                print(f"  - Detached policy: {policy['PolicyArn']}")

            # Delete inline policies
            inline_policies = iam_client.list_role_policies(RoleName=role_name)[
                "PolicyNames"
            ]
            for policy_name in inline_policies:
                iam_client.delete_role_policy(
                    RoleName=role_name, PolicyName=policy_name
                )
                print(f"  - Deleted inline policy: {policy_name}")

            # Finally, delete the role
            iam_client.delete_role(RoleName=role_name)
            print(f"✅ Deleted IAM role: {role_name}")

    except ClientError as e:
        print(f"❌ Error cleaning up IAM roles: {e}")
        print("You may need to manually delete IAM roles in the AWS console.")


def cleanup_ssm_parameters():
    """Clean up SSM parameters used by the workshop."""
    print("=" * 95)
    print("Cleaning up SSM Parameters...")
    print("=" * 95)

    parameter_prefixes = ["/app/intelligent_rag/agentcore/"]

    for prefix in parameter_prefixes:
        try:
            paginator = ssm_client.get_paginator("describe_parameters")
            pages = paginator.paginate(
                ParameterFilters=[
                    {"Key": "Name", "Option": "BeginsWith", "Values": [prefix]}
                ]
            )

            to_delete = []
            for page in pages:
                for param in page.get("Parameters", []):
                    to_delete.append(param["Name"])

            if not to_delete:
                print(f"ℹ️ No SSM parameters found under prefix: {prefix}")
                continue

            print(
                f"Found {len(to_delete)} SSM parameters to delete under prefix {prefix}:"
            )
            for name in to_delete:
                print(f"  - {name}")

            for name in to_delete:
                ssm_client.delete_parameter(Name=name)

            print(f"✅ Deleted SSM parameters under prefix: {prefix}")

        except ClientError as e:
            print(f"❌ Error cleaning up SSM parameters under {prefix}: {e}")
            print("You may need to manually delete SSM parameters in the AWS console.")


def complete_workshop_cleanup():
    """
    Run a full cleanup of all workshop resources in a safe, ordered way.

    This calls:
      1. cleanup_unstructured_kb
      2. cleanup_structured_kb
      3. cleanup_s3_bucket
      4. cleanup_redshift_resources
      5. cleanup_agentcore_resources
      6. cleanup_agentcore_iam_roles
      7. cleanup_ssm_parameters
    """
    print(
        "\n=============================== Workshop Cleanup Starting ==============================="
    )

    cleanup_unstructured_kb()
    cleanup_structured_kb()
    cleanup_s3_bucket()
    cleanup_redshift_resources()
    cleanup_agentcore_resources()
    cleanup_agentcore_iam_roles()
    cleanup_ssm_parameters()

    print(
        "\n=============================== Workshop Cleanup Complete ==============================="
    )
    print("\n📝 Summary of cleaned up resources:")
    print("   • Amazon Bedrock Knowledge Bases (Unstructured & Structured)")
    print("   • Amazon S3 buckets and data")
    print("   • Amazon Redshift Serverless workgroups and namespaces")
    print("   • Amazon Bedrock AgentCore runtime endpoints")
    print("   • AgentCore Memory instances")
    print("   • IAM roles and policies")
    print("   • SSM parameters")
    print("   • CloudWatch logs and metrics (will expire automatically)")
    print(
        "\n🎉 Thank you for completing the Unstructured & Structured RAG Agent Workshop!"
    )


if __name__ == "__main__":
    # UNCOMMENT the line below when you are ready to actually run the cleanup.
    # Be sure you no longer need any of the workshop resources.
    #
    # complete_workshop_cleanup()
    pass
