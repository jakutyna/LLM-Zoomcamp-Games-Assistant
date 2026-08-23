from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

from faq_model import FAQ

load_dotenv()
openai_client = OpenAI()
data_path = Path(__file__).resolve().parent
faq_data = pd.read_csv(data_path / "geforce_now_faq_raw.csv")

PROMPT_TEMPLATE = """
Propose a concise tag for the given GeForce NOW FAQ question.

The tag should briefly indicate what the question is about. Use no more than
4 words. Return the tag as lowercase words separated by hyphens, with no
punctuation or other characters.

FAQ id: {faq_id}
FAQ category: {category}
FAQ question: {question}
FAQ answer: {answer}
""".strip()


def process_faq(faq: pd.Series) -> dict:
    faq_id = int(faq["id"])
    question = faq["question"]
    print(f"Processing FAQ: {faq_id}. {question}")

    prompt = PROMPT_TEMPLATE.format(
        faq_id=faq_id,
        category=faq["category"],
        question=question,
        answer=faq["answer"],
    )
    response = openai_client.responses.parse(
        model="gpt-5.4-mini",
        input=[{"role": "user", "content": prompt}],
        text_format=FAQ,
    )
    return response.output_parsed.model_dump()

# Asynchronously call OpenAI API
with ThreadPoolExecutor(max_workers=5) as executor:
    results = [executor.submit(process_faq, row) for _, row in faq_data.iterrows()]

faq_with_tags = [faq.result() for faq in results]
faq_df = pd.DataFrame(faq_with_tags).sort_values("id")
faq_df.to_csv(data_path / "geforce_now_faq.csv", index=False)
