def prefilter_items(data_train):
    
    # Оставим только 5000 самых популярных товаров
    popularity = data_train.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_5000 = popularity.sort_values('n_sold', ascending=False).head(5000).item_id.tolist()
    
    #добавим, чтобы не потерять юзеров
    data_train.loc[~data_train['item_id'].isin(top_5000), 'item_id'] = 999999
    
    # В prefilter мы будем убирать из трейна только непопулярные и давно не покупаемые, чтобы не снизить метрики
    # Фильтрация по пожеланиям бизнеса (убрать очень популярные, определенных категорий, слишком дорогие и тп)
    # Будет осуществляться в postfilter, чтобы это не затронуло фазу обучения модели 
    # (учиться мы будем на всех товарах, которые реально покупают - тк они присутствуют в таргетах трейна и теста
    # и именно их мы будем использовать для вычисления метрик. 
    # А если мы исключим по требованию бизнеса из обучения товар который люди реально покупают - то он не попадет в рекомендации
    # И при вычислении метрик - это будет снижать нам скор.
    
    # Вообще, в моем понимании, фильтрацию по требованиям бизнеса нужно делать уже после разработки модели
    # Тое-сть, после того как мы поработаем с трейновыми данными, найдем все решения, максимизирующие выбранные метрики, 
    # Подберем параметры, обучим рабочую модель и утвердим ее у бизнеса... 
    # Вот уже потом - из наших, максимально точных рекомендаций, мы будем исключать те позиции, которые бизнес не хочет
    # рекомендовать по каким-то своим соображениям (и так популярны, слишком дешевы/дороги и просто потому, что не хочет и тп)
    
    # Уберем самые непопулряные 
    popularity = data_train.groupby('item_id')['user_id'].nunique().reset_index()
    popularity['user_id'] = popularity['user_id'] / data_train['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.unique().tolist()
    data_train.loc[data_train['item_id'].isin(top_notpopular), 'item_id'] = 999999
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    recent = data_train[data_train['week_no'] > data_train['week_no'].max() - 52].item_id.unique().tolist()
    data_train.loc[~data_train['item_id'].isin(recent), 'item_id'] = 999999
    
    return data_train

def postfilter_items(data, item_features, deps_to_exclude):
    
    # Эта функция будет возвращать списки item_id на исключение (для использования их в filter например)
    
    # Самые популярные 
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity['user_id'] = popularity['user_id'] / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    
    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    
    # Уберем не интересные для рекоммендаций категории (department)
    exclude_by_department = item_features[item_features['DEPARTMENT'].isin(deps_to_exclude)].PRODUCT_ID.unique().tolist()
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб. 
    cheapest = data.loc[data.price < data.price.quantile(0.2)].item_id.tolist()
    
    # Уберем слишком дорогие товары
    expensive = data.loc[data.price > data.price.quantile(0.99994)].item_id.tolist()
    
    # Самые продаваемые
    sales = data.groupby('item_id')['quantity'].sum().reset_index()
    sales.columns = ('item_id', 'sales_sum')
    top_sales = sales.loc[(items.sales_sum > 10000000)].item_id.tolist()
    
    return list(set(top_popular + exclude_by_department + cheapest + expensive + top_sales))

def get_similar_items_recommendation(data, model, user_id, N=5):
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
    
    result = top_N.lambda(x: get_similar(x))
    return result

def get_similar_users_recommendation(data, model, user_id, N=5):
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
    