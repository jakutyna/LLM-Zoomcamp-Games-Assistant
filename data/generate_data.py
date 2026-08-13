from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

from game_model import Game

load_dotenv()
openai_client = OpenAI()
data_path = Path(__file__).resolve().parent
top_games = pd.read_csv(data_path / "top_games.csv")

PROMPT_TEMPLATE = """
Generate detailed information for the given video game.

Use the game id as a rank. Provide accurate information based on real game data.
Game decription should be no longer than 300 words.

Game id: {game_id}
Game title: '{game_title}'
""".strip()

def process_game(game: pd.Series) -> dict:
    game_id = game["id"]
    game_title = game["title"]
    print(f"Processing game: {game_id}. {game_title}")

    prompt = PROMPT_TEMPLATE.format(game_id=game_id, game_title=game_title)
    response = openai_client.responses.parse(
        model="gpt-5.4-mini",
        input=[{"role": "user", "content": prompt}],
        text_format=Game,
    )
    return response.output_parsed.model_dump()

# Run API requests asynchronously
with ThreadPoolExecutor(max_workers=5) as executor:
    results = [executor.submit(process_game, row) for _, row in top_games.iterrows()]

games_data = [game.result() for game in results]
games_df = pd.DataFrame(games_data).sort_values("rank")
games_df.to_csv(data_path / "games_data.csv", index=False)
