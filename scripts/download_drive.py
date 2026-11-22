"""
Robust Google Drive Large File Downloader.
Extracts modern Google Drive download tokens, UUIDs, and confirmation forms.
"""

import os
import sys
import re
import requests


def download_google_drive_file(file_id: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    initial_url = "https://docs.google.com/uc?export=download"
    print(f"[*] Connecting to Google Drive for File ID: {file_id}...")
    res = session.get(initial_url, params={'id': file_id}, headers=headers, stream=True)

    target_url = initial_url
    target_params = {'id': file_id}
    filename = "round1.zip"

    # Check if virus scan warning / confirmation page was returned
    content_type = res.headers.get('Content-Type', '')
    if 'text/html' in content_type:
        html = res.text
        # Extract filename if visible
        fn_match = re.search(r'<a href="[^"]+">([^<]+\.(?:zip|tar|gz|pth|csv))</a>', html)
        if fn_match:
            filename = fn_match.group(1)

        # Extract form parameters
        uuid_match = re.search(r'name="uuid" value="([^"]+)"', html)
        confirm_match = re.search(r'name="confirm" value="([^"]+)"', html)

        if uuid_match:
            uuid_val = uuid_match.group(1)
            confirm_val = confirm_match.group(1) if confirm_match else 't'
            target_url = "https://drive.usercontent.google.com/download"
            target_params = {
                'id': file_id,
                'export': 'download',
                'confirm': confirm_val,
                'uuid': uuid_val
            }
            print(f"[*] Detected large file ({filename}). Submitting confirmation token...")
            res = session.get(target_url, params=target_params, headers=headers, stream=True)

    dest_file = os.path.join(dest_dir, filename)

    # Check if stream is actual file data
    if 'text/html' in res.headers.get('Content-Type', ''):
        print("[!] Error: Google Drive returned an HTML page instead of the file stream.")
        print(f"Response: {res.text[:300]}")
        return None

    total_size = int(res.headers.get('content-length', 0))
    print(f"[*] Downloading {filename} to {dest_file}...")
    if total_size > 0:
        print(f"[*] Total file size: {total_size / (1024**3):.2f} GB ({total_size / (1024**2):.1f} MB)")

    CHUNK_SIZE = 1024 * 1024  # 1 MB chunks
    downloaded = 0
    with open(dest_file, "wb") as f:
        for chunk in res.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (50 * 1024 * 1024) < CHUNK_SIZE:
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"    Downloaded: {downloaded / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB ({pct:.1f}%)")
                    else:
                        print(f"    Downloaded: {downloaded / (1024**2):.1f} MB...")

    print(f"[✓] Download completed successfully: {dest_file} ({downloaded / (1024**3):.2f} GB)")
    return dest_file


if __name__ == "__main__":
    fid = sys.argv[1] if len(sys.argv) > 1 else "1A24PTWWji509iDwf50EkJuTcwyb13RDH"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
    download_google_drive_file(fid, out_dir)
