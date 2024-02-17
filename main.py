from google_search import search_google
from relevance_feedback import get_relevance_feedback, expand_query

def main():
    initial_query = input("Enter your query: ")
    target_precision = float(input("Enter target precision@10: "))

    current_query = initial_query
    precision = 0

    while precision < target_precision:
        all_docs = search_google(current_query)
        relevant_docs = get_relevance_feedback(all_docs)

        precision = len(relevant_docs) / 10
        if precision == 0:
            print("Precision is 0. Stopping the search.")
            break

        current_query = expand_query(current_query, relevant_docs, all_docs)
        print(f"Modified query: {current_query}")

    print("Final query:", current_query)

if __name__ == "__main__":
    main()
