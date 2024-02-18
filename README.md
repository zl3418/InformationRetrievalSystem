# Information Retrieval System

## Team Members

- Zixin Li (Columbia UNI: zl3418)
- You Ji (Columbia UNI: yj2806)

## Files

```
cs6111-project/
    │
    ├── main.py                 # Main script to run the query reformulation system
    ├── google_search.py        # Module for interacting with the Google Custom Search API
    ├── requirements.txt        # Module for handling relevance feedback and query expansion
    └── utils.py                # List of Python package dependencies
    └── stopwords.txt           # List of stopwords used for query processing
    └── README.md               # Project documentation
```


## How to Run the Program

1. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

2. Run the program with the following command:

    ```
    python main.py <google_api_key> <google_engine_id> <precision> <query>
    ```


## Internal Design

- **Main Module (`main.py`)**: Orchestrates the search process, including querying Google, obtaining relevance feedback, and modifying the query based on feedback.
- **Google Search Module (`google_search.py`)**: Handles communication with the Google Custom Search API. It sends search queries to the API and processes the results to extract relevant information such as URLs, titles, and snippets of the search results.
- **Relevance Feedback Module (`relevance_feedback.py`)**: Manages the relevance feedback process and query expansion based on feedback. It presents search results to the user and collects their feedback on whether each result is relevant or not. Based on this feedback, it separates the results into relevant and non-relevant sets for further processing.
- **External Libraries**:
  - `requests`: Used for making HTTP requests to the Google Custom Search API.
  - `numpy`: Used for numerical operations in the Rocchio algorithm.
  - `scikit-learn`: Used for TF-IDF vectorization and cosine similarity calculations.
  - `nltk`: Used for natural language processing tasks.


## Query-Modification Method
- **Relevance Feedback**: After each round of search, the user provides feedback on the relevance of the retrieved documents. Based on this feedback, documents are classified into relevant and non-relevant sets.
- **Vector Representation**: The snippets of both relevant and non-relevant documents are converted into TF-IDF vectors using scikit-learn's TfidfVectorizer. This vectorization process transforms the text into a numerical representation that can be used for mathematical operations.
- **Rocchio Algorithm**: The Rocchio algorithm is applied to adjust the query vector. The algorithm computes a new query vector that is closer to the centroid of the relevant document vectors and further away from the centroid of the non-relevant document vectors. This is achieved by taking the mean of the relevant document vectors, subtracting the mean of the non-relevant document vectors, and adding this result to the original query vector.
- **Keyword Selection**: From the adjusted query vector, new keywords are selected based on their weights in the vector. The top new keywords that are not already present in the current query are chosen to be added to the query.
- **Query Word Order**: To determine the order of words in the modified query, all possible permutations of the query terms are generated. Each permutation is scored based on its average n-gram overlap with the relevant documents. The permutation with the highest score is selected as the best query for the next round.

## Google Custom Search Engine Details

- **API Key**: AIzaSyA-qNsQ7hExo8A0ZYVz7Vwq-5jIetGoF8o
- **Engine ID**: c5cfab8a2ca6a4074

## Additional Information

- Due to the recent tragic event involving the death of Susan Wojcicki's son, the Google search results have been altered, resulting in different test cases for our system. Prior to this unfortunate incident, our system consistently achieved 100% precision in identifying 23andMe co-founder Anne Wojcicki within the first iteration of search results.
