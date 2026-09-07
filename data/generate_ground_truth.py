from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from pydantic import BaseModel, Field


class GroundTruthQuestions(BaseModel):
    questions: list[str] = Field(
        min_length=5,
        max_length=5,
        description="Exactly five distinct questions answered by the FAQ record",
    )


load_dotenv()
openai_client = OpenAI()
data_path = Path(__file__).resolve().parent
faq_data = pd.read_csv(data_path / "csv" / "geforce_now_faq.csv")

PROMPT_TEMPLATE = """
Generate exactly five distinct, natural questions that a user could ask and
that the given GeForce NOW FAQ record answers. 

The questions should represent different ways a user might ask about the same
information. The output should resemble how people ask questions on the internet. 
Do not answer the questions, add numbering, or include information
that is not supported by the FAQ record.

FAQ category: {category}
FAQ tag: {tag}
FAQ question: {question}
FAQ answer: {answer}
""".strip()


def process_faq(faq: pd.Series) -> list[dict[str, int | str]]:
    faq_id = int(faq["id"])
    print(f"Processing FAQ: {faq_id}. {faq['question']}")

    prompt = PROMPT_TEMPLATE.format(
        category=faq["category"],
        tag=faq["tag"],
        question=faq["question"],
        answer=faq["answer"],
    )
    response = openai_client.responses.parse(
        model="gpt-5.4-mini",
        input=[{"role": "user", "content": prompt}],
        text_format=GroundTruthQuestions,
    )
    questions = response.output_parsed.questions
    return [{"id": faq_id, "question": question} for question in questions]


with ThreadPoolExecutor(max_workers=5) as executor:
    results = [executor.submit(process_faq, row) for _, row in faq_data.iterrows()]

ground_truth = [item for result in results for item in result.result()]
ground_truth_df = pd.DataFrame(ground_truth, columns=["id", "question"])
ground_truth_df.to_csv(data_path / "csv" / "ground_truth.csv", index=False)