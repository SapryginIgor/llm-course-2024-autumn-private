from typing import List, Optional
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sympy.codegen.fnodes import dimension

from part1.search_engine import Document, SearchResult

class FAISSSearcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Инициализация индекса
        """
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.index: Optional[faiss.Index] = None
        self.dimension: int = 384  # Размерность для 'all-MiniLM-L6-v2'


    def build_index(self, documents: List[Document]) -> None:
        """
        TODO: Реализовать создание FAISS индекса
        
        1. Сохранить документы
        2. Получить эмбеддинги через model.encode()
        3. Нормализовать векторы (faiss.normalize_L2)
        4. Создать индекс:
            - Создать quantizer = faiss.IndexFlatIP(dimension)
            - Создать индекс = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
            - Обучить индекс (train)
            - Добавить векторы (add)
        """
        self.documents = documents
        embs = []
        for doc in documents:
            text = doc.title + '\n' + doc.text
            emb = self.model.encode(text)
            embs.append(emb)
        self.embeddings = np.array(embs)
        faiss.normalize_L2(np.array(self.embeddings))
        quantizer = faiss.IndexFlatIP(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, len(documents)//39)
        index.train(self.embeddings)
        index.add(self.embeddings)
        self.index = index

        pass

    def save(self, path: str) -> None:
        """
        TODO: Реализовать сохранение индекса
        
        1. Сохранить в pickle:
            - documents
            - индекс (faiss.serialize_index)
        """
        di = {'docs': self.documents, 'faiss': self.index}
        file = open(path, 'ab')
        pickle.dump(di, file)
        file.close()
        pass

    def load(self, path: str) -> None:
        """
        TODO: Реализовать загрузку индекса
        
        1. Загрузить из pickle:
            - documents
            - индекс (faiss.deserialize_index)
        """
        file = open(path, 'rb')
        di = pickle.load(file)
        self.index = di['faiss']
        self.documents = di['docs']
        file.close()

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        TODO: Реализовать поиск
        
        1. Получить эмбеддинг запроса
        2. Нормализовать вектор
        3. Искать через index.search()
        4. Вернуть найденные документы
        """
        emb = self.model.encode(query).reshape(1,-1)
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, top_k)
        res = []
        for i, ind in enumerate(I[0]):
            doc = self.documents[ind]
            s = SearchResult(doc.id, float((D[0][i]/2)), doc.title, doc.text)
            res.append(s)
        return res

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[SearchResult]]:
        """
        TODO: Реализовать batch-поиск
        
        1. Получить эмбеддинги всех запросов
        2. Нормализовать векторы
        3. Искать через index.search()
        4. Вернуть результаты для каждого запроса
        """
        emb = self.model.encode(queries)
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, top_k)
        big_res = []
        for j in range(len(D)):
            res = []
            for i, ind in enumerate(I[j]):
                doc = self.documents[ind]
                s = SearchResult(doc.id, float((D[j][i] / 2)), doc.title, doc.text)
                res.append(s)
            big_res.append(res)
        return big_res
