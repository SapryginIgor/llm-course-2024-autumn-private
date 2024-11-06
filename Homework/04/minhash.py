import re
import pandas as pd
import numpy as np
from torch.ao.nn.quantized.functional import threshold


class MinHash:
    def __init__(self, num_permutations: int, threshold: float):
        self.num_permutations = num_permutations
        self.threshold = threshold

    def preprocess_text(self, text: str) -> str:
        return re.sub("( )+|(\n)+", " ", text).lower()

    def tokenize(self, text: str) -> set:
        text = self.preprocess_text(text)
        return set(text.split(' '))

    def get_occurrence_matrix(self, corpus_of_texts: list[set]) -> pd.DataFrame:
        '''
        Получение матрицы вхождения токенов. Строки - это токены, столбы это id документов.
        id документа - нумерация в списке начиная с нуля
        '''
        # TODO:
        di = {}
        n = len(corpus_of_texts)
        for id, text in enumerate(corpus_of_texts):
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in di:
                    di[token] = {i: 0 for i in range(n)}
                di[token][id] += 1
        df = pd.DataFrame.from_dict(di, orient='index')
        df.sort_index(inplace=True)
        return df

    def is_prime(self, a):
        if a % 2 == 0:
            return a == 2
        d = 3
        while d * d <= a and a % d != 0:
            d += 2
        return d * d > a

    def get_new_index(self, x: int, permutation_index: int, prime_num_rows: int) -> int:
        '''
        Получение перемешанного индекса.
        values_dict - нужен для совпадения результатов теста, а в общем случае используется рандом
        prime_num_rows - здесь важно, чтобы число было >= rows_number и было ближайшим простым числом к rows_number

        '''
        values_dict = {
            'a': [3, 4, 5, 7, 8],
            'b': [3, 4, 5, 7, 8]
        }
        a = values_dict['a'][permutation_index]
        b = values_dict['b'][permutation_index]
        return (a * (x + 1) + b) % prime_num_rows

    def get_minhash_similarity(self, array_a: np.array, array_b: np.array) -> float:
        '''
        Вовзращает сравнение minhash для НОВЫХ индексов. То есть: приходит 2 массива minhash:
            array_a = [1, 2, 1, 5, 3]
            array_b = [1, 3, 1, 4, 3]

            на выходе ожидаем количество совпадений/длину массива, для примера здесь:
            у нас 3 совпадения (1,1,3), ответ будет 3/5 = 0.6
        '''

        # TODO:
        return np.sum(array_a == array_b) / len(array_a)

    def get_similar_pairs(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает список из таплов индексов похожих документов, похожесть которых > threshold.
        '''
        # TODO:
        cond = np.where(self.get_similar_matrix(min_hash_matrix) > self.threshold)

        return list(filter(lambda x: x[0] != x[1], list(zip(cond[0], cond[1]))))

    def get_similar_matrix(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает матрицу расстояний
        '''
        # TODO: 
        n_cols = len(min_hash_matrix[0])
        mtx = np.zeros((n_cols, n_cols))
        for i in range(n_cols):
            for j in range(n_cols):
                mtx[i, j] = self.get_minhash_similarity(min_hash_matrix[:, i], min_hash_matrix[:, j])

        return mtx

    def find_closest_prime(self, x: int):
        a = x
        while not self.is_prime(a):
            a += 1
        return a

    def get_minhash(self, occurrence_matrix: pd.DataFrame) -> np.array:
        '''
        Считает и возвращает матрицу мин хешей. MinHash содержит в себе новые индексы. 

        new index = (2*(index +1) + 3) % 3 
        
        Пример для 1 перемешивания:
        [0, 1, 1] new index: 2
        [1, 0, 1] new index: 1
        [1, 0, 1] new index: 0

        отсортируем по новому индексу 
        [1, 0, 1]
        [1, 0, 1]
        [0, 1, 1]

        Тогда первый элемент minhash для каждого документа будет:
        Doc1 : 0
        Doc2 : 2
        Doc3 : 0
        '''
        # TODO:
        n_rows, n_cols = occurrence_matrix.shape

        prime_num_rows = self.find_closest_prime(n_rows)
        minhash_matrix = np.zeros((self.num_permutations, n_cols))
        for perm in range(self.num_permutations):
            new_index = np.zeros(n_rows)
            for i in range(n_rows):
                new_index[i] = self.get_new_index(i, perm, prime_num_rows)
            occurrence_matrix['perm'] = new_index
            permuted = occurrence_matrix.sort_values('perm')
            permuted.drop(['perm'], axis=1, inplace=True)
            permuted.index = np.arange(len(permuted))
            first_occurences = permuted.ne(0).idxmax().values
            minhash_matrix[perm] = first_occurences

        return minhash_matrix

    def run_minhash(self, corpus_of_texts: list[str]):
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash(occurrence_matrix)
        similar_pairs = self.get_similar_pairs(minhash)
        similar_matrix = self.get_similar_matrix(minhash)
        print(similar_matrix)
        return similar_pairs


class MinHashJaccard(MinHash):
    def __init__(self, num_permutations: int, threshold: float):
        self.num_permutations = num_permutations
        self.threshold = threshold

    def get_jaccard_similarity(self, set_a: set, set_b: set) -> float:
        '''
        Вовзращает расстояние Жаккарда для двух сетов. 
        '''
        # TODO:
        return len((set_a & set_b)) / len(set_a | set_b)

    def get_similar_pairs(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает список из таплов индексов похожих документов, похожесть которых > threshold.
        '''
        # TODO:
        return

    def get_similar_matrix(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает матрицу расстояний
        '''
        # TODO:

        return

    def get_minhash_jaccard(self, occurrence_matrix: pd.DataFrame) -> np.array:
        '''
        Считает и возвращает матрицу мин хешей. Но в качестве мин хеша выписываем минимальный исходный индекс, не новый.
        В такой ситуации можно будет пользоваться расстояние Жаккрада.

        new index = (2*(index +1) + 3) % 3 
        
        Пример для 1 перемешивания:
        [0, 1, 1] new index: 2
        [1, 0, 1] new index: 1
        [1, 0, 1] new index: 0

        отсортируем по новому индексу 
        [1, 0, 1] index: 2
        [1, 0, 1] index: 1
        [0, 1, 1] index: 0

        Тогда первый элемент minhash для каждого документа будет:
        Doc1 : 2
        Doc2 : 0
        Doc3 : 2
        
        '''
        # TODO:
        return

    def run_minhash(self, corpus_of_texts: list[str]):
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash_jaccard(occurrence_matrix)
        similar_pairs = self.get_similar_pairs(minhash)
        similar_matrix = self.get_similar_matrix(minhash)
        print(similar_matrix)
        return similar_pairs
