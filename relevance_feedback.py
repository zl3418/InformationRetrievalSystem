from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk import ngrams
from nltk.corpus import stopwords
nltk.download('stopwords')

def get_relevance_feedback(results):
    relevant_docs = []
    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']} - {result['link']}")
        print(result['snippet'])
        feedback = input("Relevant (Y/N)? ")
        if feedback.lower() == 'y':
            relevant_docs.append(result)
    return relevant_docs


def expand_query(current_query, relevant_docs, all_docs):
    # Preprocess and tokenize the relevant documents
    relevant_text = " ".join(doc['snippet'] for doc in relevant_docs)
    tokens = nltk.word_tokenize(relevant_text.lower())
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stopwords.words('english')]

    # Extract phrases (bigrams) from relevant documents
    bigrams = [' '.join(bigram) for bigram in ngrams(filtered_tokens, 2)]
    filtered_bigrams = [bigram for bigram in bigrams if bigram.count(' ') == 1]

    # Combine tokens and bigrams for term selection
    combined_terms = filtered_tokens + filtered_bigrams

    # Compute TF-IDF vectors for the combined terms and all documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([current_query] + combined_terms + [doc['snippet'] for doc in all_docs])
    query_vector = tfidf_matrix[0]
    term_vectors = tfidf_matrix[1:len(combined_terms)+1]

    # Compute cosine similarity between the query vector and each term vector
    similarity_scores = cosine_similarity(query_vector, term_vectors)[0]

    # Rank the terms based on similarity scores
    ranked_terms = [term for _, term in sorted(zip(similarity_scores, combined_terms), key=lambda pair: pair[0], reverse=True)]

    # Select top new terms that are not already in the current query, considering individual words in bigrams
    current_query_terms = set(current_query.split())
    new_terms = []
    for term in ranked_terms:
        if len(new_terms) >= 2:
            break
        words = term.split()
        for word in words:
            if word not in current_query_terms and len(new_terms) < 2:
                new_terms.append(word)
                current_query_terms.add(word)

    # Add new terms to the current query
    expanded_query = " ".join(current_query.split() + new_terms)

    return expanded_query



