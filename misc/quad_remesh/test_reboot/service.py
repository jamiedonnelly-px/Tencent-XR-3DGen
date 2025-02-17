import os
import time
import subprocess

LOG_FILE = "/mnt/aigc_bucket_4/sz/mount_ok.txt"

while True:
    if not os.path.exists(LOG_FILE):
        print("File not found, restarting system...")
        subprocess.run(["sudo", "reboot"])
        break
    time.sleep(60)