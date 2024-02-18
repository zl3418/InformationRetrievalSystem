from itertools import permutations
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk import ngrams


def read_stopwords():
    """Returns list of stopwords."""
    f = open("stopwords.txt", "r")
    words = f.read().split()
    f.close()
    print("words")
    print(words)
    return words


def ngram_overlap_score(permutation, relevant_docs, n=2):
    # Tokenize and generate n-grams for the permutation, converting to lowercase
    perm_tokens = permutation.lower().split()
    if len(perm_tokens) < n:
        return 0  # Return zero score if permutation is too short for n-grams
    perm_ngrams = set(ngrams(perm_tokens, n))

    # Generate n-grams for each relevant document and calculate overlap
    overlap_scores = []
    for doc in relevant_docs:
        doc_tokens = doc.get('snippet', '').lower().split()
        if len(doc_tokens) < n:
            continue  # Skip document if it's too short for n-grams
        doc_ngrams = set(ngrams(doc_tokens, n))
        overlap = len(perm_ngrams & doc_ngrams) / len(perm_ngrams)
        overlap_scores.append(overlap)

    # Return the average overlap score, or zero if no scores were calculated
    return sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0


def rocchio_optimal_query(relevant_doc_vectors, non_relevant_doc_vectors):
    # Compute the average vector for relevant and non-relevant documents
    avg_relevant_vector = np.mean(relevant_doc_vectors, axis=0) if relevant_doc_vectors.size > 0 else np.zeros_like(
        relevant_doc_vectors.shape[1])
    avg_non_relevant_vector = np.mean(non_relevant_doc_vectors,
                                      axis=0) if non_relevant_doc_vectors.size > 0 else np.zeros_like(
        non_relevant_doc_vectors.shape[1])

    # Compute the optimal query vector
    optimal_query_vector = avg_relevant_vector - avg_non_relevant_vector
    return optimal_query_vector


def expand_query(current_query, relevant_docs, non_relevant_docs, all_docs):
    # Compute TF-IDF vectors for all documents
    corpus = [doc.get('snippet', '') for doc in all_docs]
    tfidf_vectorizer = TfidfVectorizer(stop_words=read_stopwords())
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus).toarray()

    # Extract the TF-IDF vectors for relevant and non-relevant documents
    relevant_vectors = tfidf_matrix[[all_docs.index(doc) for doc in relevant_docs]]
    non_relevant_vectors = tfidf_matrix[[all_docs.index(doc) for doc in non_relevant_docs]]

    # Compute the Rocchio optimal query vector
    optimal_query_vector = rocchio_optimal_query(relevant_vectors, non_relevant_vectors)

    # Rank terms based on their values in the optimal query vector
    term_values = [(i, value) for i, value in enumerate(optimal_query_vector)]
    sorted_term_values = sorted(term_values, key=lambda x: x[1], reverse=True)
    ranked_terms = [tfidf_vectorizer.get_feature_names_out()[i] for i, _ in sorted_term_values]

    # Select top new terms that are not already in the current query
    current_query_terms = set(current_query.split())
    new_terms = []
    for term in ranked_terms:
        if term not in current_query_terms:
            new_terms.append(term)
            if len(new_terms) >= 2:
                break

    # Generate all possible permutations of the query terms
    all_query_terms = current_query_terms.union(new_terms)
    permutations_scores = {}
    for perm in permutations(all_query_terms):
        perm_query = " ".join(perm)
        score = ngram_overlap_score(perm_query, relevant_docs)
        permutations_scores[perm_query] = score

    # Select the permutation with the highest cosine similarity score as the best order
    best_query = max(permutations_scores, key=permutations_scores.get)

    return best_query, new_terms


def get_relevance_feedback(results):
    relevant_docs = []
    non_relevant_docs = []
    print("Google Search Results:")
    print("======================")
    for i, result in enumerate(results, start=1):
        print(f"Result {i}")
        print("[")
        print(f" URL: {result.get('link', 'No URL available')}")
        print(f" Title: {result.get('title', 'No title available')}")
        print(f" Summary: {result.get('snippet', 'No summary available')}")
        print("]")
        feedback = input(f"Relevant (Y/N)? ")
        if feedback.lower() == 'y':
            relevant_docs.append(result)
        else:
            non_relevant_docs.append(result)
    print("======================")
    return [relevant_docs, non_relevant_docs]
