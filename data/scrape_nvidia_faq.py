"""
Scrape NVIDIA GeForce NOW FAQ page and save to CSV.
"""

import csv
import re
import requests
from bs4 import BeautifulSoup


def clean_text(element) -> str:
    """
    Extract text from an element while preserving spaces between inline elements.

    Args:
        element: BeautifulSoup element to extract text from

    Returns:
        Cleaned text with proper spacing
    """
    if not element:
        return ""

    # Get text with separator to preserve spaces between elements
    text = element.get_text(separator=" ", strip=True)

    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def scrape_nvidia_faq(url: str) -> list[dict]:
    """
    Scrape FAQ data from NVIDIA GeForce NOW FAQ page.

    Args:
        url: The URL of the FAQ page

    Returns:
        List of dictionaries containing FAQ data
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    faq_data = []
    faq_id = 1
    current_category = "General"

    # Parse h2 (categories) and h3 (questions) structure
    all_headers = soup.find_all(["h2", "h3"])

    for header in all_headers:
        if header.name == "h2":
            current_category = clean_text(header)
        elif header.name == "h3":
            question = clean_text(header)

            # Find the answer by looking through next siblings
            answer = ""
            next_elem = header.find_next_sibling()
            answer_parts = []

            while next_elem and next_elem.name not in ["h2", "h3"]:
                if next_elem.name in ["p", "div", "ul", "ol"]:
                    answer_parts.append(clean_text(next_elem))
                next_elem = next_elem.find_next_sibling()

            answer = " ".join(answer_parts)

            if question and question not in ["GeForce NOW FAQs"]:
                faq_data.append({
                    "id": faq_id,
                    "category": current_category,
                    "question": question,
                    "answer": answer if answer else "Answer not available"
                })
                faq_id += 1

    return faq_data


def save_to_csv(data: list[dict], filename: str) -> None:
    """
    Save FAQ data to a CSV file.

    Args:
        data: List of FAQ dictionaries
        filename: Output CSV filename
    """
    if not data:
        print("No data to save!")
        return

    fieldnames = ["id", "category", "question", "answer"]

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Saved {len(data)} FAQ items to {filename}")


def main():
    url = "https://www.nvidia.com/en-us/geforce-now/faq/"
    output_file = "nvidia_geforce_now_faq.csv"

    print(f"Scraping FAQ data from {url}...")
    faq_data = scrape_nvidia_faq(url)

    if faq_data:
        print(f"Found {len(faq_data)} FAQ items")
        save_to_csv(faq_data, output_file)
    else:
        print("No FAQ data found. The page may require JavaScript rendering.")
        print("Consider using Selenium or Playwright for dynamic content.")


if __name__ == "__main__":
    main()
