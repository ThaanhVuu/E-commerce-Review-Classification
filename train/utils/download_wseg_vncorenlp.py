import os
import urllib.request

def download_vncorenlp(save_dir):
    jar_path   = os.path.join(save_dir, 'VnCoreNLP-1.2.jar')
    wseg_dir   = os.path.join(save_dir, 'models', 'wordsegmenter')

    if os.path.exists(jar_path) and os.path.exists(wseg_dir):
        print(f"VnCoreNLP đã tồn tại tại {save_dir}, bỏ qua download.")
        return

    print("Đang download VnCoreNLP...")
    os.makedirs(wseg_dir, exist_ok=True)

    files = {
        jar_path: "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.2.jar",
        os.path.join(wseg_dir, 'vi-vocab'):         "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab",
        os.path.join(wseg_dir, 'wordsegmenter.rdr'): "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr",
    }

    for dest, url in files.items():
        if os.path.exists(dest):
            print(f"  Đã có: {os.path.basename(dest)}")
            continue
        print(f"  Đang tải: {os.path.basename(dest)}...")
        urllib.request.urlretrieve(url, dest)

    print("Download hoàn tất.")