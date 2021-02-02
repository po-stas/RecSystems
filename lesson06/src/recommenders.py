import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

import os, sys

module_path = os.path.abspath(os.path.join(os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from src.utils import prefilter_items

class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, n_factors=70,
                               top_5000=False,
                               strip_not_popular=False,
                               strip_outdated=False,
                               weighting=True,  
                               K1=100,
                               B=0.5): 
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        self.data = prefilter_items(data, top_5000, strip_not_popular, strip_outdated)
        
        self.user_item_matrix = self.prepare_matrix(self.data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T, K1=K1, B=B).T 
        
        self.model = self.fit(self.user_item_matrix, n_factors=n_factors)
        # self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        
        def to_item_id(idx):
            return self.id_to_itemid[idx]

        self.v_to_item_id = np.vectorize(to_item_id)
    
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
    
    
    @staticmethod
    def prepare_matrix(data):
        
        user_item_matrix = pd.pivot_table(data, 
                                  index='user_id', columns='item_id', 
                                  values='quantity', 
                                  aggfunc='count', 
                                  fill_value=0
                                 )
        
        return user_item_matrix.astype(float) 
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        # (csr_matrix(user_item_matrix).T.tocsr() > 0) * 1
        
        return model

    def get_popularity_recommendations(self, N=5):
        """Топ-N популярных товаров"""
    
        popular = self.data.groupby('item_id')['sales_value'].sum().reset_index()
        popular.sort_values('sales_value', ascending=False, inplace=True)

        recs = popular.head(N).item_id

        return recs.tolist()
    
    def get_all_recommendations(self, user_ids, N=5):
        """Выдает рекомендации для всех пользователей датафрейма user_ids. N - требуемое количество рекомендаций"""

        fast_recs = self.model.user_factors @ self.model.item_factors.T
        recs = user_ids.apply(lambda x: self.v_to_item_id(np.argsort(fast_recs[self.userid_to_id[x]])[-N:]) if
                             x in self.userid_to_id.keys() else self.get_popularity_recommendations(N))
        
        # Если их меньше N то дополняем из популярити рекомендаций
        return recs.apply(lambda x: x if len(x)==N else x + self.get_popularity_recommendations(N - len(x)))
        
    
    def get_recommendations(self, user_id, N=5):
        """Выдает рекомендации для пользователя user_id. N - требуемое количество рекомендаций"""
        
        fast_recs = self.model.user_factors @ self.model.item_factors.T
        return self.v_to_item_id(np.argsort(fast_recs[self.userid_to_id[user_id]])[-N:])
        

    def get_similar_items_recommendation(self, user_id, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        def get_similar(item_id, N=2):
            recs = self.model.similar_items(self.itemid_to_id[item_id], N)
            return self.id_to_itemid[recs[1][0]]
    
        popularity = self.data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)

        # Top N товаров популярных у этого пользователя
        top_N = popularity[popularity['user_id']==user_id].item_id.values[:N]

        # Пока без векторизации.. надо проверить как это будет работать
        res = [get_similar(item_id) for item_id in top_N]

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_similar_users_recommendation(self, user_id, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
        # Похожие пользователи
        sim_users = self.model.similar_users(self.userid_to_id[user_id], N)

        # Популярные товары для похожих пользователей
        popularity = self.data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)

        res = popularity[popularity['user_id'].isin(sim_users)].item_id.values[:N]

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res