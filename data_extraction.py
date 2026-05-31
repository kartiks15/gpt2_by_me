import os
import requests

file_path = "train_data.txt"

should_download = not os.path.exists(file_path)

if not should_download:
    with open(file_path, "r", encoding="utf-8") as file:
        should_download = "<!DOCTYPE html" in file.read(200)

if should_download:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": "Adolf Hitler",
    }

    headers = {
        "User-Agent": "gpt2scratch/1.0 (local training data script)"
    }

    response = requests.get(url, params=params, headers=headers, timeout=10)

    if response.status_code == 403:
        print("Access denied. Try a different User-Agent or URL.")
    else:
        response.raise_for_status()
        data = response.json()
        pages = data["query"]["pages"]
        article_text = next(iter(pages.values()))["extract"]

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(article_text)

        print("Saved training data to train_data.txt")
