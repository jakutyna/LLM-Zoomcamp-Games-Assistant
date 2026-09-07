import os


DEFAULT_ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
DEFAULT_INDEX = "geforce-now-faq"
DEFAULT_VECTOR_INDEX = "geforce-now-faq-vector"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BEST_BOOST_PARAMS = {"question": 3, "tag": 2, "answer": 3}
