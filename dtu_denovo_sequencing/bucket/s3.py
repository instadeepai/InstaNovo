import os
from pathlib import Path
from typing import Optional

from cloudpathlib import S3Client
from cloudpathlib import S3Path

from dtu_denovo_sequencing.utils.io import Openable


class S3BucketManager:
    """Python interface to download / upload files and directories from / to s3 bucket.

    Attributes:
        default_bucket: name of the default bucket to use.
    """

    def __init__(self, default_bucket: Optional[str] = None) -> None:
        self.default_bucket = default_bucket
        self.set_client()

    @staticmethod
    def set_client() -> None:
        """Set the S3 client.

        Raises:
            OSError: if there is a missing env variable:
                - AWS_SECRET_ACCESS_KEY
                - AWS_ACCESS_KEY_ID
                - S3_ENDPOINT
        """
        try:
            aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
            aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
            endpoint = os.environ["S3_ENDPOINT"]
        except KeyError:
            msg = (
                "To use the S3 bucket you must export the following env variables: "
                "AWS_SECRET_ACCESS_KEY, AWS_ACCESS_KEY_ID and S3_ENDPOINT"
            )
            raise OSError(msg)

        S3Client(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=f"https://{endpoint}",
        ).set_as_default_client()

    def upload(
        self,
        local_path: Openable,
        bucket_path: str,
        force: bool = False,
        bucket_name: Optional[str] = None,
    ) -> str:
        """Upload a file or a directory to the bucket path provided.

        Use cases:
            - local_path + bucket_path are files
            - local_path + bucket_path are directories
            - local_path is a file and bucket_path a directory: the local filename is kept

        Args:
            local_path: path to the local source file / directory to upload
            bucket_path: path to the bucket target file / directory.
            force: indicates if the file / directory is erased in case it already exists in
                the bucket.
            bucket_name: optional name of the bucket to use.

        Returns: path to the file / directory uploaded in the bucket.

        Raises:
            FileNotFoundError: local_path provided does not exist
        """
        local_path = Path(local_path)

        if not local_path.exists():
            raise FileNotFoundError(
                f"The local file / directory {local_path} does not exist."
            )

        s3_path = self._build_s3_path(bucket_path, bucket_name)

        # in case we upload a file to a directory, we need to add the file name because
        # S3Path does not know this is a directory if it doesn't exist yet
        if local_path.is_file() and Path(s3_path).suffix == "":
            s3_path /= local_path.name

        s3_path.upload_from(local_path, force_overwrite_to_cloud=force)

        return str(s3_path)

    def download(
        self,
        local_path: Openable,
        bucket_path: str,
        force: bool = False,
        bucket_name: Optional[str] = None,
    ) -> str:
        """Download a file or a directory to the local path provided.

        Use cases:
            - local_path + bucket_path are files
            - local_path + bucket_path are directories
            - bucket_path is a file and local_path a directory: the bucket filename is kept

        Args:
            local_path: path to the local target file / directory
            bucket_path: path to the bucket source file / directory to download
            force: indicates if the file / directory is erased in case it already exists
                locally.
            bucket_name: optional name of the bucket to use.

        Returns: path to the local file / directory

        Raises:
            FileExistsError: the local_path already exist and force=False
        """
        local_path = Path(local_path)

        if not force and local_path.exists():
            raise FileExistsError(
                f"The local file / directory {local_path}. Use force=True to overwrite it."
            )

        # Due to a bug with cloudpathlib (https://github.com/drivendataorg/cloudpathlib/issues/57)
        # we ensure that the parent directory of local_file_path (or itself if it is a directory)
        # is created
        local_directory = local_path.parent if local_path.suffix else local_path
        local_directory.mkdir(exist_ok=True, parents=True)

        s3_path = self._build_s3_path(bucket_path, bucket_name)
        s3_path.download_to(local_path)

        # In case the local_path provided is a directory, append the filename
        # to return the full path
        if local_path.is_dir() and not s3_path.is_dir():
            local_path /= s3_path.name

        return str(local_path)

    def remove(self, bucket_path: str, bucket_name: Optional[str] = None) -> None:
        """Remove a file / directory from the bucket.

        Args:
            bucket_file_path: path to the file / directory to delete on the bucket.
            bucket_name: name of the bucket to use.
        """
        s3_path = self._build_s3_path(bucket_path, bucket_name)

        if s3_path.is_dir():
            s3_path.rmtree()

        elif s3_path.is_file():
            s3_path.unlink()

    def _build_s3_path(self, bucket_path: str, bucket_name: Optional[str]) -> S3Path:
        """Build the S3Path to use."""
        if bucket_path.startswith(S3Path.cloud_prefix):
            return S3Path(bucket_path)

        bucket = bucket_name or self.default_bucket

        if bucket is None:
            msg = (
                "The bucket to use cannot be retrieved. If the bucket is not available in "
                "the bucket_path you must provide the bucket_name or define the default_bucket"
                " to use by the class."
            )
            raise ValueError(msg)

        s3_path = S3Path(f"{S3Path.cloud_prefix}{bucket}/{bucket_path}")

        return s3_path
