def get_relevance_feedback(results):
    relevant_docs = []
    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']} - {result['link']}")
        print(result['snippet'])
        feedback = input("Relevant (Y/N)? ")
        if feedback.lower() == 'y':
            relevant_docs.append(result)
    return relevant_docs

def expand_query(current_query, relevant_docs):
    # Dummy implementation - replace with your actual query expansion logic
    new_query = current_query + " additional_keyword"
    return new_query
