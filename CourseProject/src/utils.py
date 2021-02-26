import numpy as np

def prefilter_items(data_train, items_to_filter=[], N=0, top_N=None, strip_not_popular=True, strip_outdated=True, strip_not_selling=True, strip_cheapest=1.0):
    
    result = data_train.copy()
    
    # Убираем товары из списка для фильтрации (Какие-нибудь дополнительныет требования)
    result.loc[result['item_id'].isin(items_to_filter), 'item_id'] = 999999
    
    # Убираем top N по популярности
    if N > 0:
        popularity = data_train.groupby('item_id')['user_id'].nunique().reset_index()
        popularity['user_id'] = popularity['user_id'] / data_train['user_id'].nunique()
        popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

        top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
        
        result.loc[result['item_id'].isin(top_popular[:N]), 'item_id'] = 999999
    
    # Оставим только 5000 самых популярных товаров
    if top_N:
        popularity = data_train.groupby('item_id')['quantity'].sum().reset_index()
        popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
        only_top_N = popularity.sort_values('n_sold', ascending=False).head(top_N).item_id.tolist()
    
        #добавим, чтобы не потерять юзеров
        result.loc[~result['item_id'].isin(only_top_N), 'item_id'] = 999999
    
    # Уберем самые непопулряные 
    if strip_not_popular:
        popularity = data_train.groupby('item_id')['user_id'].nunique().reset_index()
        popularity['user_id'] = popularity['user_id'] / data_train['user_id'].nunique()
        popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
        top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.unique().tolist()
        result.loc[result['item_id'].isin(top_notpopular), 'item_id'] = 999999
        
    # Продавшиеся менее 50 раз
    if strip_not_selling:
        popularity = data_train.groupby('item_id')['quantity'].sum().reset_index()
        popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
        bottom_50 = popularity.sort_values('n_sold', ascending=True).head(50).item_id.tolist()
        result.loc[result['item_id'].isin(bottom_50), 'item_id'] = 999999
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    if strip_outdated:
        recent = data_train[data_train['week_no'] > data_train['week_no'].max() - 52].item_id.unique().tolist()
        result.loc[~result['item_id'].isin(recent), 'item_id'] = 999999
        
    if strip_cheapest:
        data_train['price'] = data_train['sales_value'] / data_train['quantity']
        cheapest = data_train.loc[data_train.price < strip_cheapest].item_id.tolist()
        result.loc[result['item_id'].isin(cheapest), 'item_id'] = 999999
    
    return result

def postfilter_items(data, item_features, deps_to_exclude=[], N=3, filter_cheapest=True, most_sold=False):
    
    """Эта функция будет возвращать списки item_id на исключение (для использования их в filter например)"""
    
    # Самые популярные 
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity['user_id'] = popularity['user_id'] / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    
    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()[:N]
    
    # Уберем не интересные для рекоммендаций категории (department)
    exclude_by_department = item_features[item_features['department'].isin(deps_to_exclude)].item_id.unique().tolist()
    
    # Уберем слишком дешевые товары (<1$). 1 покупка из рассылок стоит 60 руб. 
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    cheapest = data.loc[data.price < data.price.quantile(0.1)].item_id.tolist()
    if not filter_cheapest:
        cheapest = []
    
    # Уберем слишком дорогие товары
    # data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    # expensive = data.loc[data.price > data.price.quantile(0.99994)].item_id.tolist()
    
    # Самые продаваемые
    sales = data.groupby('item_id')['quantity'].sum().reset_index()
    sales.columns = ('item_id', 'sales_sum')
    top_sales = sales.loc[(sales.sales_sum > 10000000)].item_id.tolist()
    if not most_sold:
        top_sales = []
    
    return list(set(top_popular + exclude_by_department + top_sales + cheapest))


def get_similar_items_recommendation(data, model, user_id, itemid_to_id, id_to_itemid, N=5):
    '''get_similar_items_recommendation(transaction_data, model, user_id, number_of_similar_items)'''
    
    def get_similar(item_id, N=2):
        recs = model.similar_items(itemid_to_id[item_id], N)
        return id_to_itemid[recs[1][0]]
    
    # Получить 5 наиболее похожих товаров для тех, которые пользователь больше всего покупает
    # Функция принимает user_id 
    # Возвращает top N similar item_ids для максимально покупаемых юзером
    
    popularity = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
    popularity.sort_values('quantity', ascending=False, inplace=True)
    
    # Top N товаров популярных у этого пользователя
    top_N = popularity[popularity['user_id']==user_id].item_id.values[:N]
    
    result = [get_similar(item_id) for item_id in top_N]
    return result

def get_similar_users_recommendation(data, model, user_id, userid_to_id, N=5):
    '''get_similar_users_recommendation(transaction_data, model, user_id, number_of_similar_items)
    '''
    
    # Получить тоp N товаров которые нравятся похожим пользователям
    # Похожие пользователи
    sim_users = model.similar_users(userid_to_id[user_id], N)
    
    # Популярные товары для похожих пользователей
    popularity = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
    popularity.sort_values('quantity', ascending=False, inplace=True)
    
    top_N = popularity[popularity['user_id'].isin(sim_users)].item_id.values[:N]
    
    return top_N
    