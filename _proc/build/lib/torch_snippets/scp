import os
from pathlib import Path as P

import paramiko
from loguru import logger
from scp import SCPClient as SCP


class SCPClient:
    def __init__(
        self, hostname, port, username, password=None, private_key=None, logfile=None
    ):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.private_key = private_key
        self.client = None
        logfile = "/tmp/{time}-scp.log" if logfile is None else logfile
        self.logger = logger
        self.logger.add(logfile)

    def connect(self):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if self.private_key:
            self.client.connect(
                self.hostname,
                port=self.port,
                username=self.username,
                key_filename=self.private_key,
            )
        else:
            self.client.connect(
                self.hostname,
                port=self.port,
                username=self.username,
                password=self.password,
            )

    def close(self):
        if self.client:
            self.client.close()

    def upload(self, local_path, remote_path):
        self.connect()
        try:
            with SCP(self.client.get_transport()) as scp:
                if os.path.isdir(local_path):
                    scp.put(local_path, recursive=True, remote_path=remote_path)
                else:
                    scp.put(local_path, remote_path)
                self.logger.info(f"Uploaded {local_path} to {remote_path}")
        except Exception as e:
            self.logger.warning(f"Error uploading: {e}")
        finally:
            self.close()

    def download(self, remote_path, local_path):
        self.connect()
        try:
            with SCP(self.client.get_transport()) as scp:
                is_remote_dir = "." not in remote_path
                not_a_dir = "" if is_remote_dir else "not"
                self.logger.info(f"Assuming {remote_path} is {not_a_dir}a directory")
                if is_remote_dir:
                    scp.get(remote_path, recursive=True, local_path=local_path)
                else:
                    os.makedirs(P(local_path).parent, exist_ok=True)
                    scp.get(remote_path, local_path)
                self.logger.info(f"Downloaded {remote_path} to {local_path}")
        except Exception as e:
            self.logger.warning(f"Error downloading: {e}")
        finally:
            self.close()


# Example usage
if __name__ == "__main__":
    hostname = "10.161.141.73"
    port = 22  # Default port for SSH
    username = "jioaidev"
    password = os.environ["JIOAIDEV_PASSWORD"]
    private_key = None  # Or specify the path to your private key file

    scp_client = SCPClient(hostname, port, username, password, private_key)
    local_path = "/tmp/tmp.csv"
    remote_path = "/data/datasets/210-invoices/051--2.5k-invoices-20231027/ToCleanup/vitstr_80k/00010100002900891622023/0.csv"

    # Upload a file/folder
    # scp_client.upload(local_path, remote_path)

    # Or download a file/folder
    scp_client.download(remote_path, local_path)
