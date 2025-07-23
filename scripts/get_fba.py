import hashlib, os, sys, time, urllib.request

URLS = [
    "https://github.com/MarcoForte/FBA_Matting/releases/download/v1.0/fba_matting.pth",
    # Mirror (HuggingFace): uncomment if first fails
    # "https://huggingface.co/KenjiT/FBA-Matting/resolve/main/fba_matting.pth",
]

DEST = "models/fba_matting.pth"
CHUNK = 1024 * 1024
EXPECTED_MIN_BYTES = 100_000_000  # ~100MB sanity check
SHA256 = "0e3fb13d1c2d6f7f6e3c57b7e3c1e3b6d0e9a7b2d0a1e3a1f6a3d2b1c6c17c1"  # put real hash if you want

def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def download(url):
    print(f"Downloading from: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(DEST+".part", "wb") as f:
        while True:
            chunk = r.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
    os.replace(DEST+".part", DEST)

def main():
    os.makedirs("models", exist_ok=True)
    for i, url in enumerate(URLS, 1):
        try:
            download(url)
        except Exception as e:
            print(f"[{i}] Failed: {e}")
            time.sleep(2)
            continue
        size = os.path.getsize(DEST)
        print("Size:", size, "bytes")
        if size < EXPECTED_MIN_BYTES:
            print("File too small, trying next mirror...")
            continue
        # Optional checksum validation (comment out if you don't care)
        # if sha256sum(DEST) != SHA256:
        #     print("Checksum mismatch, trying next mirror...")
        #     continue
        print("✅ Download OK:", DEST)
        return
    print("❌ All mirrors failed.")
    sys.exit(1)

if __name__ == "__main__":
    main()