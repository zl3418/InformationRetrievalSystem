from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

def get_relevance_feedback(results):
    relevant_docs = []
    for i, result in enumerate(results):
        print(f"{i + 1}. {result['title']} - {result['link']}")
        print(result['snippet'])
        feedback = input("Relevant (Y/N)? ")
        if feedback.lower() == 'y':
            relevant_docs.append(result)
    return relevant_docs


def expand_query(current_query, relevant_docs):
    # Tokenize and preprocess the relevant documents
    relevant_text = " ".join(doc['snippet'] for doc in relevant_docs)
    tokens = nltk.word_tokenize(relevant_text.lower())
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stopwords.words('english')]

    # Calculate term frequency
    term_freq = Counter(filtered_tokens)

    # Select top 2 terms for expansion
    new_terms = [term for term, _ in term_freq.most_common(2)]

    # Add new terms to the current query
    expanded_query = " ".join([current_query] + new_terms)
    return expanded_query
