import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

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
                               items_to_filter=[],
                               top_5000=False,
                               strip_not_popular=False,
                               strip_outdated=False,
                               weighting=True,  
                               K1=100,
                               B=0.5): 
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        self.data = prefilter_items(data, items_to_filter, top_5000, strip_not_popular, strip_outdated)
        
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

    
class SecondLayerRecommender:
    """Модель второго уровня (Классификатор, Ранкер)
    
    Input
    -----
    data: pd.DataFrame
        Матрица взаимодействий user-item
    first_layer_recs: pd.DataFrame
        Рекомендации из модели первого уровня
    first_layer_model: MainRecommender
        Обученная модель первого уровня
    N: int
        Количество кандидатов из модели первого уровня
    user_features: pd.DataFrame
        Данные юзеров
    user_features: pd.DataFrame
        Данные айтемов 
    cat_features: Tuple
        список категориальных фичей
    to_filter: list
        Item_ids которые нужно убрать из предсказаний (требования бизнеса)
    fe_lambdas: function
        Набор действий над данными (Feature Engineering)
    iterations: int
    rate: float
    """
    
    def __init__(self, data, first_layer_model, N, user_features, item_features, cat_features, to_filter=None, fe_lambdas=None, iterations=50, rate=0.1): 
        
        self.user_features = user_features
        self.item_features = item_features
        self.first_model = first_layer_model
        self.fe_lambdas = fe_lambdas
        self.to_filter = to_filter if to_filter is not None else []
        
        self.N = N
        
        self.data = self.prepare_dataset(data)
        
        #TODO: feature ingineering here
        
        self.cat_features = cat_features
        self.model = self.fit(iterations, rate)

    @staticmethod
    def prepare_table(data, matrix, targets=True):
        
        df=pd.DataFrame({'user_id':data.user_id.values.repeat(len(data.als[0])),
                 'item_id':np.concatenate(data.als.values)})
        result = df.copy()
        if targets:
            result = matrix[['user_id', 'item_id']].copy()
            result['target'] = 1  # тут только покупки 
            result = df.merge(result, on=['user_id', 'item_id'], how='left')
            result['target'].fillna(0, inplace=True)
        
        return result
    
    @staticmethod
    def append_features(data, features, on_field):
        return data.merge(features, on=on_field, how='left')
    
    def append_embeddings(self, data):
        embs = pd.DataFrame(self.first_model.model.item_factors)
        embs['item_id'] = embs.index
        embs['item_id'] = embs['item_id'].apply(lambda x: self.first_model.id_to_itemid[x])

        result = data.merge(embs, on='item_id', how='left')
        
        embs = pd.DataFrame(self.first_model.model.user_factors)
        embs['user_id'] = embs.index
        embs['user_id'] = embs['user_id'].apply(lambda x: self.first_model.id_to_userid[x])

        return result.merge(embs, on='user_id', how='left')
    
    def prepare_dataset(self, data, targets=True):
        
        result = pd.DataFrame(data.user_id.unique())
        result.columns = ('user_id',)
        result['als'] = self.first_model.get_all_recommendations(result['user_id'], N=self.N)
        
        result = self.prepare_table(result, data, targets=targets)
        
        if self.user_features is not None:
            result = self.append_features(result, self.user_features, on_field='user_id')
        
        if self.item_features is not None:
            result = self.append_features(result, self.item_features, on_field='item_id')
        
        result = self.append_embeddings(result)
        try:
            result['manufacturer'] = result['manufacturer'].fillna(0).astype('int')
        except KeyError:
            pass
        
        if self.fe_lambdas is not None:
            result = fe_lambdas(result)
        
        return result.fillna(0)
    
    
    def fit(self, iterations=50, learning_rate=0.1, loss='Logloss'):
        """Обучает CatBoost"""
        
        X = self.data.drop('target', axis=1)
        y = self.data[['target']]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
        
        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            loss_function=loss,
            custom_loss=['AUC', 'Accuracy']
        )
        
        model.fit(
            X_train, y_train,
            cat_features=self.cat_features,
            eval_set=(X_test, y_test),
            plot=True,
            verbose=False
        )
        
        return model
    
    def get_all_recommendations(self, user_ids, N=5):
        """Вычислить N рекомендаций для юзеров из списка user_ids"""
        
        test_df = pd.DataFrame(user_ids)
        test_df.columns = ('user_id',)
        
        test_df = self.prepare_dataset(test_df, targets=False)
        
        preds = self.model.predict_proba(test_df)
        
        # Мерджим полученные вероятности с таблицей user_id - item_id
        preds_df = pd.DataFrame(preds[:,1:])
        preds_df.columns = ('probabilities',)
        prob_df = test_df.join(preds_df)
        
        # Группируем айтемы по юзеру сортируя по убыванию probability
        sorted_prob = prob_df.sort_values('probabilities', ascending=False).reset_index(drop=True)
        result = sorted_prob.groupby('user_id')['item_id'].unique().reset_index()
        
        result.columns = ('user_id', 'recommendations')
        result['recommendations'] = result['recommendations'].apply(lambda x: x[:N])
        return result
        
