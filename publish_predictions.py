"""
publish_predictions.py

Publishes ML predictions to various destinations:
- Azure Blob Storage (partitioned parquet)
- SQL Server (optional)
- Notifications (Teams/Slack/Email) (optional)

Usage:
    python publish_predictions.py --input predictions.parquet --config config/settings.json
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pyodbc
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionPublisher:
    """Publishes predictions to multiple destinations."""
    
    def __init__(self, config_path: str = "config/settings.json"):
        """Initialize publisher with configuration."""
        self.config = self._load_config(config_path)
        load_dotenv()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def publish(self, predictions_df: pd.DataFrame, timestamp: Optional[datetime] = None):
        """
        Publish predictions to all configured destinations.
        
        Args:
            predictions_df: DataFrame with predictions
            timestamp: Timestamp for partitioning (default: now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        logger.info("="*80)
        logger.info("PUBLISHING PREDICTIONS")
        logger.info("="*80)
        logger.info(f"Timestamp: {timestamp}")
        logger.info(f"Predictions: {len(predictions_df)} rows")
        
        # 1. Publish to Blob Storage
        if self.config['output']['enable_blob_storage']:
            self._publish_to_blob(predictions_df, timestamp)
        
        # 2. Publish to SQL Server
        if self.config['output']['enable_sql_publish']:
            self._publish_to_sql(predictions_df)
        
        # 3. Send notifications
        if self.config['output']['enable_notifications']:
            self._send_notifications(predictions_df, timestamp)
        
        logger.info("✓ Publishing complete")
    
    def _publish_to_blob(self, df: pd.DataFrame, timestamp: datetime):
        """Publish predictions to Azure Blob Storage as partitioned parquet."""
        logger.info("\n[1/3] Publishing to Azure Blob Storage")
        
        try:
            # Get output path from environment or config
            output_base = os.environ.get(
                'AZUREML_OUTPUT_predictions',
                self.config['storage']['predictions_path']
            )
            
            # Create partition path: YYYY/MM/DD/HH/predictions.parquet
            partition_format = self.config['output']['prediction_partitioning']
            if partition_format == "YYYY/MM/DD/HH":
                partition_path = f"{timestamp.year:04d}/{timestamp.month:02d}/{timestamp.day:02d}/{timestamp.hour:02d}"
            else:
                partition_path = timestamp.strftime("%Y/%m/%d/%H")
            
            # Full output path
            if output_base.startswith('azureml://'):
                # Running in Azure ML
                output_path = output_base.replace(
                    'azureml://datastores/workspaceblobstore/paths/',
                    '/mnt/batch/tasks/shared/LS_root/mounts/clusters/workspaceblobstore/'
                )
            else:
                output_path = output_base
            
            full_path = os.path.join(output_path, partition_path)
            os.makedirs(full_path, exist_ok=True)
            
            # Write parquet
            output_file = os.path.join(full_path, "predictions.parquet")
            df.to_parquet(output_file, index=False, engine='pyarrow')
            
            logger.info(f"  ✓ Saved to: {output_file}")
            logger.info(f"  Size: {os.path.getsize(output_file) / 1024:.2f} KB")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to publish to blob: {str(e)}")
            raise
    
    def _publish_to_sql(self, df: pd.DataFrame):
        """Publish predictions to SQL Server table."""
        logger.info("\n[2/3] Publishing to SQL Server")
        
        try:
            # Get SQL connection details
            server = os.getenv('SQL_SERVER')
            database = os.getenv('SQL_DATABASE')
            username = os.getenv('SQL_USERNAME')
            password = os.getenv('SQL_PASSWORD')
            
            if not all([server, database, username, password]):
                logger.warning("  ⚠ SQL credentials not found, skipping SQL publish")
                return
            
            # Connect to SQL Server
            conn_str = (
                f"DRIVER={{ODBC Driver 18 for SQL Server}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"UID={username};"
                f"PWD={password};"
                f"Encrypt=yes;"
                f"TrustServerCertificate=no;"
            )
            
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            
            # Get table name from config
            table_name = self.config['sql']['predictions_table']
            batch_size = self.config['sql']['batch_size']
            
            # Create table if not exists
            create_table_sql = f"""
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{table_name.split('.')[-1].strip('[]')}')
            CREATE TABLE {table_name} (
                PredictionID INT IDENTITY(1,1) PRIMARY KEY,
                TimestampHour DATETIME NOT NULL,
                SettlementPoint VARCHAR(100),
                DART_Prediction FLOAT,
                ModelType VARCHAR(50),
                PredictionTimestamp DATETIME DEFAULT GETUTCDATE(),
                INDEX IX_TimestampHour (TimestampHour),
                INDEX IX_SettlementPoint (SettlementPoint)
            )
            """
            cursor.execute(create_table_sql)
            conn.commit()
            
            # Insert predictions in batches
            insert_sql = f"""
            INSERT INTO {table_name} (TimestampHour, SettlementPoint, DART_Prediction, ModelType)
            VALUES (?, ?, ?, ?)
            """
            
            rows_inserted = 0
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                values = [
                    (
                        row.get('TimestampHour'),
                        row.get('SettlementPoint'),
                        row.get('DART_Prediction'),
                        row.get('ModelType', 'lgbm')
                    )
                    for _, row in batch.iterrows()
                ]
                
                cursor.executemany(insert_sql, values)
                conn.commit()
                rows_inserted += len(batch)
                
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"  Inserted {rows_inserted:,} / {len(df):,} rows...")
            
            cursor.close()
            conn.close()
            
            logger.info(f"  ✓ Inserted {rows_inserted:,} predictions to {table_name}")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to publish to SQL: {str(e)}")
            # Don't raise - SQL publishing is optional
    
    def _send_notifications(self, df: pd.DataFrame, timestamp: datetime):
        """Send notifications about predictions."""
        logger.info("\n[3/3] Sending Notifications")
        
        try:
            # Calculate summary statistics
            avg_prediction = df['DART_Prediction'].mean()
            min_prediction = df['DART_Prediction'].min()
            max_prediction = df['DART_Prediction'].max()
            std_prediction = df['DART_Prediction'].std()
            
            message = f"""
📊 ERCOT DART Predictions Generated

⏰ Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC
📈 Predictions: {len(df):,} records

📉 Statistics:
  • Average: ${avg_prediction:.2f}/MWh
  • Min: ${min_prediction:.2f}/MWh
  • Max: ${max_prediction:.2f}/MWh
  • Std Dev: ${std_prediction:.2f}/MWh

✅ Predictions published successfully
            """.strip()
            
            # Send to Teams
            teams_url = self.config['notifications'].get('teams_webhook_url')
            if teams_url:
                self._send_teams_message(teams_url, message)
            
            # Send to Slack
            slack_url = self.config['notifications'].get('slack_webhook_url')
            if slack_url:
                self._send_slack_message(slack_url, message)
            
            logger.info("  ✓ Notifications sent")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to send notifications: {str(e)}")
    
    def _send_teams_message(self, webhook_url: str, message: str):
        """Send message to Microsoft Teams."""
        import requests
        
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "summary": "ERCOT Predictions",
            "themeColor": "0078D7",
            "title": "📊 ERCOT DART Predictions",
            "text": message
        }
        
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
    
    def _send_slack_message(self, webhook_url: str, message: str):
        """Send message to Slack."""
        import requests
        
        payload = {
            "text": message
        }
        
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Publish ML predictions')
    parser.add_argument('--input', type=str, required=True, help='Input predictions parquet file')
    parser.add_argument('--config', type=str, default='config/settings.json', help='Configuration file')
    parser.add_argument('--timestamp', type=str, help='Timestamp for partitioning (YYYY-MM-DD HH:MM:SS)')
    
    args = parser.parse_args()
    
    try:
        logger.info("="*80)
        logger.info("PREDICTION PUBLISHER")
        logger.info("="*80)
        
        # Load predictions
        logger.info(f"Loading predictions from: {args.input}")
        df = pd.read_parquet(args.input)
        logger.info(f"  Loaded {len(df):,} predictions")
        
        # Parse timestamp
        if args.timestamp:
            timestamp = datetime.strptime(args.timestamp, '%Y-%m-%d %H:%M:%S')
        else:
            timestamp = datetime.utcnow()
        
        # Publish
        publisher = PredictionPublisher(args.config)
        publisher.publish(df, timestamp)
        
        logger.info("\n" + "="*80)
        logger.info("✓ PUBLISHING COMPLETE")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Publishing failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

