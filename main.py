from google_search import search_google
from relevance_feedback import get_relevance_feedback, expand_query
import sys


def main():
    if len(sys.argv) != 5:
        print("Usage: python Main.py <google api key> <google engine id> <precision> <query>")
        sys.exit(1)

    google_api_key = sys.argv[1]
    google_engine_id = sys.argv[2]
    try:
        target_precision = float(sys.argv[3])
        if not (0 < target_precision <= 1):
            raise ValueError("Precision must be between 0 and 1.")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    initial_query = sys.argv[4]

    current_query = initial_query
    precision = 0

    while precision < target_precision:
        print("Parameters:")
        print(f"Client key  = {google_api_key}")
        print(f"Engine key  = {google_engine_id}")
        print(f"Query       = {current_query}")
        print(f"Precision   = {target_precision}")
        all_docs = search_google(current_query, google_api_key, google_engine_id)
        feedback_results = get_relevance_feedback(all_docs)
        relevant_docs = feedback_results[0]
        non_relevant_docs = feedback_results[1]

        precision = len(relevant_docs) / 10
        print("FEEDBACK SUMMARY")
        print(f"Query {current_query}")
        print(f"Precision {precision}")
        if precision == 0:
            print("Precision is 0. Stopping the search.")
            break
        elif precision >= target_precision:
            print(f"Desired precision reached, done")
            break
        else:
            print(f"Still below the desired precision of {target_precision}")

        best_query, new_terms = expand_query(current_query, relevant_docs, non_relevant_docs, all_docs)
        print(f"Augmenting by: {' '.join(new_terms)}")
        current_query = best_query


if __name__ == "__main__":
    main()
