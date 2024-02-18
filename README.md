# Information Retrieval System

## Team Members

- Zixin Li (Columbia UNI: zl3418)
- You Ji (Columbia UNI: yj2806)

## Files Submitted

- `Main.py`: The main script to run the program.
- `google_search.py`: Module for interacting with the Google Custom Search API.
- `relevance_feedback.py`: Module for handling relevance feedback and query expansion.
- `requirements.txt`: List of Python package dependencies.
- `stopwords.txt`: List of stopwords used for query processing.
- `README.md`: This file, containing project documentation.

## How to Run the Program

1. Unzip the code repository and navigate into cs6111-project

    ```
    cd cs6111-project
    ```

3. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

4. Run the program with the following command:

    ```
    python Main.py <google_api_key> <google_engine_id> <precision> <query>
    ```

## Internal Design

- **Main Module (`Main.py`)**: Orchestrates the search process, including querying Google, obtaining relevance feedback, and modifying the query based on feedback.
- **Google Search Module (`google_search.py`)**: Handles communication with the Google Custom Search API.
- **Relevance Feedback Module (`relevance_feedback.py`)**: Manages the relevance feedback process and query expansion based on feedback.
- **External Libraries**:
  - `requests`: Used for making HTTP requests to the Google Custom Search API.
  - `numpy`: Used for numerical operations in the Rocchio algorithm.
  - `scikit-learn`: Used for TF-IDF vectorization and cosine similarity calculations.
  - `nltk`: Used for natural language processing tasks.

## Query-Modification Method

- **Keyword Selection**: In each round, we use the Rocchio algorithm to adjust the query vector based on relevance feedback. We then select the top new keywords that are not already in the current query.
- **Query Word Order**: We generate all possible permutations of the query terms and rank them based on their average n-gram overlap with the relevant documents. The permutation with the highest score is chosen as the best query for the next round.

## Google Custom Search Engine Details

- **API Key**: AIzaSyA-qNsQ7hExo8A0ZYVz7Vwq-5jIetGoF8o
- **Engine ID**: c5cfab8a2ca6a4074
