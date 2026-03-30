from pathlib import Path

import requests
from bs4 import BeautifulSoup

SOURCE_URL = "https://en.wikibooks.org/wiki/Cognitive_Psychology_and_Cognitive_Neuroscience/Decision_Making_and_Reasoning"
OUTPUT_FILE = Path(__file__).resolve().with_name("Selected_Document.txt")


def extract_text_from_webpage(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
    except requests.RequestException as exc:
        print(f"Failure: request could not be completed. {exc}")
        return ""

    if response.status_code != 200:
        print(f"Failure: HTTP status code {response.status_code}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")

    cleaned_paragraphs = []
    for paragraph in paragraphs:
        text = paragraph.get_text(strip=True)
        if text:
            cleaned_paragraphs.append(text)

    extracted_text = "\n\n".join(cleaned_paragraphs)
    OUTPUT_FILE.write_text(extracted_text, encoding="utf-8")

    print(f"Success: HTTP status code {response.status_code}")
    return extracted_text


def main() -> None:
    extract_text_from_webpage(SOURCE_URL)


if __name__ == "__main__":
    main()