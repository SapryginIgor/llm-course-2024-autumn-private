from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import json
import pickle

from sentence_transformers import SentenceTransformer

@dataclass
class Document:
   id: str
   title: str
   text: str
   embedding: Optional[np.ndarray] = None

@dataclass
class SearchResult:
   doc_id: str
   score: float
   title: str
   text: str

def load_documents(path: str) -> List[Document]:
   """Загрузка документов из json файла"""
   with open(path, 'r', encoding='utf-8') as f:
       data = json.load(f)
   return [
       Document(
           id=article['id'],
           title=article['title'],
           text=article['text'],
           embedding=None
       )
       for article in data['articles']
   ]

class Indexer:
   def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
       self.model = SentenceTransformer(model_name)
       self.documents: List[Document] = []
       self.embeddings: Optional[np.ndarray] = None

   def add_documents(self, documents: List[Document]) -> None:
       """
       TODO: Реализовать индексацию документов
       1. Сохранить документы в self.documents
       2. Получить эмбеддинги для документов используя self.model.encode()
          Подсказка: для каждого документа нужно объединить title и text
       3. Сохранить эмбеддинги в self.embeddings
       """
       self.documents = documents
       embs = []
       for doc in documents:
           text = doc.title +'\n'+doc.text
           emb = self.model.encode(text)
           embs.append(emb)
       self.embeddings = np.array(embs)


   def save(self, path: str) -> None:
       """
       TODO: Реализовать сохранение индекса
       1. Сохранить self.documents и self.embeddings в pickle файл
       """
       di = {'docs':self.documents, 'embs':self.embeddings}
       file = open(path, 'ab')
       pickle.dump(di, file)
       file.close()
       pass

   def load(self, path: str) -> None:
       """
       TODO: Реализовать загрузку индекса
       1. Загрузить self.documents и self.embeddings из pickle файла
       """
       file = open(path, 'rb')
       di = pickle.load(file)
       self.documents = di['docs']
       self.embeddings = di['embs']
       file.close()

class Searcher:
   def __init__(self, index_path: str, model_name: str = 'all-MiniLM-L6-v2'):
       """
       TODO: Реализовать инициализацию поиска
       1. Загрузить индекс из index_path
       2. Инициализировать sentence-transformers
       """
       self.indexer = Indexer(model_name)
       self.indexer.load(index_path)
       self.model = SentenceTransformer(model_name)
       pass

   def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
       """
       TODO: Реализовать поиск документов
       1. Получить эмбеддинг запроса через self.model.encode()
       2. Вычислить косинусное сходство между запросом и документами
       3. Вернуть top_k наиболее похожих документов
       """

       emb = self.model.encode(query)
       norms = np.linalg.norm(self.indexer.embeddings, axis=1)
       emb_norm = np.linalg.norm(emb)
       dist = np.arccos((self.indexer.embeddings @ emb)/ (norms*emb_norm))/np.pi
       best = np.argpartition(dist, top_k)[:top_k]
       res = []
       for ind in best:
           doc = self.indexer.documents[ind]
           s = SearchResult(doc.id, float((dist[ind])), doc.title, doc.text)
           res.append(s)
       return res
       pass
