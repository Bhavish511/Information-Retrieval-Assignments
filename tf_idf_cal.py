import os
import numpy as np
import math
import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def apply_transformations(word):
    transformations = [
        (r'\b(\d+)\b', ''),  
        (r'\b(\w)\1+\b', r'\1'),  
        (r'[-–]', ''),  
        (r'[\[\](){};:,/“”]', ''),  
        (r'\b\w*(\d+)\w*\b', ''),  
        (r'\b\w{16,}\b', ''),  
    ]
    for pattern, replacement in transformations:
        word = re.sub(pattern, replacement, word)
    return word

def process_documents(text, stop_words):
    stemmed_words = []
    words = re.findall(r'\b\w+\b', text)
    for token in words:
        token_lower = token.lower()
        token_stemmed = stemmer.stem(token_lower)
        token_transformed = apply_transformations(token_stemmed).lower()
        if (token_transformed not in stop_words) and (len(token_transformed) > 1) and (len(token_transformed) < 16):
            stemmed_words.append(token_transformed)
    return stemmed_words

def build_tfidf(vsm_index, docs, idf):
    doc_tfidf = {}
    for doc in docs:
        file = os.path.join('ResearchPapers', str(doc) + '.txt')
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            filtered_text = process_documents(text, stop_words)
            for idx, term in enumerate(vsm_index):
                term_frequency = filtered_text.count(term)
                if doc not in doc_tfidf:
                    doc_tfidf[doc] = np.zeros(len(vsm_index))
                doc_tfidf[doc][idx] = np.around(term_frequency * idf[term], 3)

        e_len = np.sqrt(np.sum(np.square(doc_tfidf[doc])))
        doc_tfidf[doc] = np.around(doc_tfidf[doc] / e_len, 3)
        doc_tfidf[doc][np.isnan(doc_tfidf[doc])] = 0
    
    return doc_tfidf

def save_tfidf_results(doc_tfidf):
    with open('tfidf_results.txt', 'w') as f:
        for doc, tfidf_vector in doc_tfidf.items():
            f.write(f'Document: {doc}\n')
            f.write(f'TF-IDF Vector: {",".join(map(str, tfidf_vector))}\n')


if __name__ == "__main__":
    stop_words = set()
    with open('Stopword-List.txt', 'r') as f:
        stop_words = set(f.read().split())

    Invertindex = {}  # Initialize Inverted Index Dictionary
    ResearchPapers = [1, 2, 3, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26]  # List of Research Papers to process
    docs = np.array(ResearchPapers)  # Initialize empty array to store document IDs
    doc_indices = {doc_id: idx for idx, doc_id in enumerate(docs)}  # Create a mapping between document IDs and indices in the docs array

    for doc in ResearchPapers:
        with open(f"ResearchPapers/{doc}.txt", "r") as f:
            paper = f.read()

        words = re.findall(r'\b\w+\b', paper)

        for token in words:
            token_lower = token.lower()
            token_stemmed = stemmer.stem(token_lower)
            token_transformed = apply_transformations(token_stemmed).lower()

            if (token_transformed not in stop_words) and (len(token_transformed) > 1) and (len(token_transformed) < 16):
                if token_transformed in Invertindex:
                    loc = doc_indices[doc]
                    Invertindex[token_transformed][loc] += 1
                else:
                    temp_doc_vector = np.zeros(len(docs), dtype=int)
                    loc = doc_indices[doc]
                    temp_doc_vector[loc] += 1
                    Invertindex[token_transformed] = temp_doc_vector.astype(np.int32)

    DF = {}
    for term, doc_vector in Invertindex.items():
        DF[term] = np.count_nonzero(doc_vector)

    IDF = {}
    total_docs = len(docs)
    for term, df in DF.items():
        IDF[term] = math.log10(total_docs / df)

    vsm_index = list(Invertindex.keys())
    doc_tfidf = build_tfidf(vsm_index, docs, IDF)
    save_tfidf_results(doc_tfidf)

    
