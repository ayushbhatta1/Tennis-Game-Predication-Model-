"""
Download extra Sackmann data: point-by-point repositories.
These use the same player IDs as the existing historical CSVs.
"""

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")


REPOS = [
    ("https://github.com/JeffSackmann/tennis_pointbypoint.git", "pointbypoint"),
    ("https://github.com/JeffSackmann/tennis_slam_pointbypoint.git", "slam_pointbypoint"),
]


def download_repo(url, target_name):
    target = os.path.join(DATA_DIR, target_name)
    if os.path.exists(target):
        print(f"  {target_name}: already exists, pulling updates...")
        subprocess.run(["git", "-C", target, "pull", "--quiet"], check=False)
    else:
        print(f"  {target_name}: cloning...")
        os.makedirs(DATA_DIR, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", url, target], check=True)
    csv_count = sum(1 for f in os.listdir(target) if f.endswith(".csv"))
    print(f"  {target_name}: {csv_count} CSV files")


def main():
    print("Downloading extra Sackmann data...")
    for url, name in REPOS:
        try:
            download_repo(url, name)
        except Exception as e:
            print(f"  ERROR downloading {name}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
