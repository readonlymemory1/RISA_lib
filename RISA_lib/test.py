import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def find_sentence_similarity(text, target_word):
    # 문장 토큰화
    sentences = sent_tokenize(text)
    
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # 타겟 단어 벡터화
    target_word_tfidf = vectorizer.transform(target_word)
    
    # 코사인 유사도 계산
    cosine_similarities = cosine_similarity(target_word_tfidf, tfidf_matrix).flatten()
    
    # 유사도와 함께 문장 반환
    sentences_with_similarity = list(zip(sentences, cosine_similarities))
    sentences_with_similarity.sort(key=lambda x: x[1], reverse=True)  # 유사도 기준으로 정렬
    return sentences_with_similarity

# 예제 텍스트
text = """
Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant indentation. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.
Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly, procedural), object-oriented and functional programming. Python is often described as a "batteries included" language due to its comprehensive standard library.
"""

# 찾고자 하는 단어
target_word = ["Python", "programming"]

# 단어와 문장 사이의 연관도 측정
sentences_with_similarity = find_sentence_similarity(text, target_word)

# 결과 출력
for sentence, similarity in sentences_with_similarity:
    print(f"Similarity: {similarity:.4f} - Sentence: {sentence}")
