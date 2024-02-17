from google_search import search_google
from relevance_feedback import get_relevance_feedback, expand_query

def main():
    initial_query = input("Enter your query: ")
    target_precision = float(input("Enter target precision@10: "))

    current_query = initial_query
    precision = 0

    while precision < target_precision:
        print("Parameters:")
        print(f"Client key  = YOUR_CLIENT_KEY")
        print(f"Engine key  = YOUR_ENGINE_KEY")
        print(f"Query       = {current_query}")
        print(f"Precision   = {target_precision}")
        all_docs = search_google(current_query)
        relevant_docs = get_relevance_feedback(all_docs)

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

        new_query = expand_query(current_query, relevant_docs, all_docs)
        print(f"Augmenting by {' '.join(new_query.split()[len(current_query.split()):])}")
        current_query = new_query

    print("Final query:", current_query)

if __name__ == "__main__":
    main()
