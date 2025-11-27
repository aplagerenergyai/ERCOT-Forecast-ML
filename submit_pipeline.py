"""
submit_pipeline.py

Helper script to submit the ERCOT training pipeline to Azure ML.
Provides easy command-line interface and progress monitoring.
"""

import argparse
import sys
import time
from datetime import datetime

try:
    from azure.ai.ml import MLClient
    from azure.ai.ml import load_job
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml.entities import PipelineJob
except ImportError:
    print("Error: Azure ML SDK not installed")
    print("Install with: pip install azure-ai-ml azure-identity")
    sys.exit(1)


def get_ml_client(subscription_id: str, resource_group: str, workspace_name: str) -> MLClient:
    """Create Azure ML client."""
    credential = DefaultAzureCredential()
    
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    
    print(f"✓ Connected to workspace: {workspace_name}")
    return ml_client


def submit_pipeline(ml_client: MLClient, pipeline_file: str) -> PipelineJob:
    """Submit the training pipeline."""
    print(f"\nSubmitting pipeline: {pipeline_file}")
    
    # Load pipeline YAML
    pipeline_job = load_job(source=pipeline_file)
    
    # Submit job
    job = ml_client.jobs.create_or_update(pipeline_job)
    
    print(f"✓ Pipeline submitted successfully")
    print(f"  Job name: {job.name}")
    print(f"  Status: {job.status}")
    print(f"  Studio URL: {job.studio_url}")
    
    return job


def monitor_pipeline(ml_client: MLClient, job_name: str, poll_interval: int = 30):
    """Monitor pipeline execution."""
    print(f"\nMonitoring pipeline: {job_name}")
    print(f"Polling every {poll_interval} seconds (Ctrl+C to stop monitoring)\n")
    
    try:
        while True:
            job = ml_client.jobs.get(job_name)
            status = job.status
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Status: {status}")
            
            # Check if job is complete
            if status in ["Completed", "Failed", "Canceled"]:
                print(f"\n{'='*60}")
                print(f"Pipeline {status.upper()}")
                print(f"{'='*60}")
                
                if status == "Completed":
                    print("\n✓ All models trained successfully!")
                    print(f"\nOutputs:")
                    if hasattr(job, 'outputs'):
                        for output_name, output_val in job.outputs.items():
                            print(f"  - {output_name}: {output_val}")
                    return True
                elif status == "Failed":
                    print("\n✗ Pipeline failed. Check logs in Azure ML Studio.")
                    return False
                else:
                    print("\n⚠ Pipeline was canceled.")
                    return False
            
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        print("\n\nStopped monitoring (pipeline continues running in background)")
        print(f"Check status at: {job.studio_url}")
        return None


def list_pipeline_runs(ml_client: MLClient, limit: int = 10):
    """List recent pipeline runs."""
    print(f"\nRecent pipeline runs (last {limit}):\n")
    
    jobs = ml_client.jobs.list()
    
    count = 0
    for job in jobs:
        if count >= limit:
            break
        
        if job.type == "pipeline":
            print(f"Name: {job.name}")
            print(f"  Status: {job.status}")
            print(f"  Created: {job.creation_context.created_at}")
            print(f"  URL: {job.studio_url}")
            print()
            count += 1


def main():
    parser = argparse.ArgumentParser(
        description="Submit and monitor ERCOT training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit pipeline and monitor
  python submit_pipeline.py --submit
  
  # Submit without monitoring
  python submit_pipeline.py --submit --no-monitor
  
  # Monitor existing pipeline
  python submit_pipeline.py --monitor <job-name>
  
  # List recent pipeline runs
  python submit_pipeline.py --list
        """
    )
    
    parser.add_argument(
        '--subscription-id',
        type=str,
        required=False,
        help='Azure subscription ID'
    )
    parser.add_argument(
        '--resource-group',
        type=str,
        required=False,
        help='Azure resource group name'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        required=False,
        help='Azure ML workspace name'
    )
    parser.add_argument(
        '--submit',
        action='store_true',
        help='Submit the training pipeline'
    )
    parser.add_argument(
        '--monitor',
        type=str,
        metavar='JOB_NAME',
        help='Monitor an existing pipeline job'
    )
    parser.add_argument(
        '--no-monitor',
        action='store_true',
        help='Submit without monitoring (use with --submit)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List recent pipeline runs'
    )
    parser.add_argument(
        '--pipeline-file',
        type=str,
        default='aml_training_pipeline.yml',
        help='Path to pipeline YAML file (default: aml_training_pipeline.yml)'
    )
    parser.add_argument(
        '--poll-interval',
        type=int,
        default=30,
        help='Polling interval in seconds (default: 30)'
    )
    
    args = parser.parse_args()
    
    # If no action specified, show help
    if not any([args.submit, args.monitor, args.list]):
        parser.print_help()
        sys.exit(0)
    
    # Get Azure credentials
    if not all([args.subscription_id, args.resource_group, args.workspace]):
        print("Please provide Azure credentials:")
        print("  --subscription-id <id>")
        print("  --resource-group <name>")
        print("  --workspace <name>")
        print("\nOr set environment variables:")
        print("  AZURE_SUBSCRIPTION_ID")
        print("  AZURE_RESOURCE_GROUP")
        print("  AZURE_WORKSPACE_NAME")
        sys.exit(1)
    
    # Create ML client
    try:
        ml_client = get_ml_client(
            subscription_id=args.subscription_id,
            resource_group=args.resource_group,
            workspace_name=args.workspace
        )
    except Exception as e:
        print(f"Error connecting to Azure ML: {e}")
        sys.exit(1)
    
    # Execute requested action
    if args.list:
        list_pipeline_runs(ml_client)
    
    elif args.submit:
        job = submit_pipeline(ml_client, args.pipeline_file)
        
        if not args.no_monitor:
            monitor_pipeline(ml_client, job.name, args.poll_interval)
    
    elif args.monitor:
        monitor_pipeline(ml_client, args.monitor, args.poll_interval)


if __name__ == "__main__":
    main()

