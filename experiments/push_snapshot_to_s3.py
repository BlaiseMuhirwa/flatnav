import boto3
import os
import sys
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)


def upload_file_to_s3(file_path: str, bucket_name: str) -> None:
    """
    Uploads a file to an s3 bucket.
    :param file_path: The path to the file to upload.
    :param bucket_name: The name of the bucket to upload to.
    """
    s3_client = boto3.client("s3")
    try:
        s3_client.upload_file(file_path, bucket_name, os.path.basename(file_path))
        logger.info(
            f"Uploaded {file_path} to s3://{bucket_name}/{os.path.basename(file_path)}"
        )

    except Exception as e:
        logger.error(
            f"Failed to upload {file_path} to s3://{bucket_name}/{os.path.basename(file_path)}"
        )
        logger.error(e)


def find_most_recent_file(directory: str, file_extension: str) -> str:
    """
    Finds the most recent file in a directory with a given file extension.
    :param directory: The directory to search.
    :param file_extension: The file extension to search for.
    :return: The path to the most recent file.
    """
    latest_time = datetime.min
    latest_file = None

    for file in os.listdir(directory):
        if file.endswith(file_extension):
            file_path = os.path.join(directory, file)
            filetime = datetime.strptime(file, "index_%a %b %d %H:%M:%S %Y.hnsw")

            if filetime > latest_time:
                latest_time = filetime
                latest_file = file_path

    return latest_file


if __name__ == "__main__":
    directory = "/root"
    bucket_name = "hnsw-index-snapshots"
    file_extension = ".hnsw"

    latest_file = find_most_recent_file(directory, file_extension)

    if latest_file:
        try:
            upload_file_to_s3(latest_file, bucket_name)
            os.remove(latest_file)
            logger.info(f"deleted file {latest_file} after uploading to s3")
        except Exception:
            logger.error(
                f"Failed to upload file {latest_file} to s3 bucket {bucket_name}",
                exc_info=True,
            )
