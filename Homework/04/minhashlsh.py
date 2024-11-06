import pandas as pd
import numpy as np
from sympy import are_similar

from minhash import MinHash

class MinHashLSH(MinHash):
    def __init__(self, num_permutations: int, num_buckets: int, threshold: float):
        self.num_permutations = num_permutations
        self.num_buckets = num_buckets
        self.threshold = threshold
        
    def get_buckets(self, minhash: np.array) -> np.array:
        '''
        Возвращает массив из бакетов, где каждый бакет представляет собой N строк матрицы сигнатур.
        '''
        # TODO:
        n = len(minhash)
        n_buckets = min(self.num_buckets, self.num_permutations)
        r = n % n_buckets
        bucket_size = len(minhash) // n_buckets
        n_cols = minhash.shape[1]
        buckets = list(minhash[:n-r].reshape((n-r)//bucket_size, bucket_size, n_cols))
        if r != 0:
            buckets.append(minhash[n-r:])
        return buckets
    
    def get_similar_candidates(self, buckets) -> list[tuple]:
        '''
        Находит потенциально похожих кандижатов.
        Кандидаты похожи, если полностью совпадают мин хеши хотя бы в одном из бакетов.
        Возвращает список из таплов индексов похожих документов.
        '''
        # TODO:
        n_docs = len(buckets[0][0])
        are_similar = np.zeros((n_docs, n_docs), dtype=bool)
        for bucket in buckets:
            for i in range(n_docs):
                for j in range(n_docs):
                    if are_similar[i][j] or i == j:
                        continue
                    are_similar[i][j] = (self.get_minhash_similarity(bucket[:,i],bucket[:,j]) == 1)

        cond = np.where(are_similar)
        similar_candidates = list(zip(cond[0], cond[1]))


        return similar_candidates
        
    def run_minhash_lsh(self, corpus_of_texts: list[str]) -> list[tuple]:
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash(occurrence_matrix)
        buckets = self.get_buckets(minhash)
        similar_candidates = self.get_similar_candidates(buckets)
        
        return set(similar_candidates)
    
