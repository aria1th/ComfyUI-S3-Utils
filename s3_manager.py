import os
from dotenv import load_dotenv
import boto3
from botocore.config import Config

# Load environment variables
load_dotenv()

# Get AWS credentials from environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
# Get configuration settings from environment variables (optional)
READ_TIMEOUT = int(os.getenv('AWS_READ_TIMEOUT', '240'))
CONNECT_TIMEOUT = int(os.getenv('AWS_CONNECT_TIMEOUT', '120'))
MAX_RETRIES = int(os.getenv('AWS_MAX_RETRIES', '5'))

# Configure the client with timeouts and retry settings
my_config = Config(
    read_timeout=READ_TIMEOUT,
    connect_timeout=CONNECT_TIMEOUT,
    retries={'max_attempts': MAX_RETRIES}
)

# Create S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
    config=my_config
)

def test():
    # List S3 buckets
    response = s3_client.list_buckets()
    print("S3 bucket access success")

test()
