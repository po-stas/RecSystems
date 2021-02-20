import pandas as pd
import numpy as np

# Монструозная лямбда - для рассчета user-item фичей внутри класса рекоммендер
# Когда у нас будут уже предложенные взаимодействия - user-item мы можем из матрицы интеракций достать более
# специфическую информацию о паре юзер-айтем - но пары user-item у нас формируются внутри класса,
# мы подадим в класс closure, которая будет работать уже по объединенной таблице 
def closure(user_feats_with_dept):

    def fe_lambdas(data):
        d = data.copy()
        d.department = d.department.apply(lambda x: x if x != 0 else 'GROCERY')
        d['mean_minus_price'] = d.mean_value - d.price
        d['mean_user_bought_div_mean_item_bought'] = (d.mean_quantity_per_month/4) / (d.mean_quantity_per_week)


        user_feats_with_dept.mean_by_dept.fillna(0, inplace=True)
        
        # Средний чек юзера, сумма трат и количество покупок для департамента в котором находится item
        merge_w_mean = d[['user_id', 'item_id', 'department']].merge(user_feats_with_dept[['user_id', 'department', \
                                                                                           'mean_by_dept', 'sum_by_dept', \
                                                                                           'quantity_by_dept']],
                                                   on=['user_id', 'department'], how='left')

        mean_by_item_dept = merge_w_mean.groupby(['user_id', 
                                                  'item_id', 
                                                  'department'])['mean_by_dept', 'sum_by_dept', \
                                                                 'quantity_by_dept'].first().reset_index()
        
        mean_by_item_dept.columns = mean_by_item_dept.columns[:3].to_list() \
        + ['mean_in_item_dept', 'sum_in_item_dept', 'quantity_in_item_dept']
        
        d = d.merge(mean_by_item_dept[['user_id', 'item_id', 'mean_in_item_dept', 'sum_in_item_dept', 'quantity_in_item_dept']],
                    on=['user_id', 'item_id'], how='left')
        
        d['mean_in_item_dept_minus_price'] = d.mean_in_item_dept - d.price
        d['mean_in_item_dept_minus_item_mean_for_dept'] = d.mean_in_item_dept - d.item_mean_by_dept
        d['sum_in_item_dept_minus_item_sum_for_dept'] = d.sum_in_item_dept - d.item_sum_by_dept
        d['quantity_in_item_dept_minus_item_quantity_for_dept'] = d.quantity_in_item_dept - d.item_quantity_by_dept

        
        # Средний чек юзера/в неделю, сумма трат и количество покупок для департамента в котором находится item
        merge_w_mean = d[['user_id', 'item_id', 'department']].merge(user_feats_with_dept[['user_id', 'department', \
                                                                                           'mean_by_dept_per_week', \
                                                                                           'sum_by_dept_per_week', \
                                                                                           'quantity_by_dept_per_week']],
                                                   on=['user_id', 'department'], how='left')

        mean_by_item_dept_pw = merge_w_mean.groupby(['user_id', 
                                                  'item_id', 
                                                  'department'])['mean_by_dept_per_week', 'sum_by_dept_per_week', \
                                                                 'quantity_by_dept_per_week'].first().reset_index()
        
        mean_by_item_dept_pw.columns = mean_by_item_dept_pw.columns[:3].to_list() \
        + ['mean_in_item_dept_pw', 'sum_in_item_dept_pw', 'quantity_in_item_dept_pw']
        
        d = d.merge(mean_by_item_dept_pw[['user_id', 'item_id', 'mean_in_item_dept_pw', 'sum_in_item_dept_pw', \
                                          'quantity_in_item_dept_pw']], on=['user_id', 'item_id'], how='left')
        
        mean_by_item_dept_pw = mean_by_item_dept_pw.merge(d.groupby('department')['mean_in_item_dept_pw'].mean(), 
                                                          on='department', how='left')
        
        mean_by_item_dept_pw.columns = mean_by_item_dept_pw.columns[:6].to_list() \
        + ['mean_in_item_dept_pw_for_all_users']
        
        d = d.merge(mean_by_item_dept_pw[['user_id', 'item_id', 'mean_in_item_dept_pw_for_all_users']], 
                    on=['user_id', 'item_id'], how='left')
        
        d['mean_in_item_dept_per_week_minus_price'] = d.mean_in_item_dept_pw - d.price
        d['mean_in_item_dept_per_week_minus_mean_of_all_users_for_dept'] = d.mean_in_item_dept_pw - \
            d.mean_in_item_dept_pw_for_all_users

     
        
        
        # Средний чек юзера, сумма трат и количество покупок для commodity к которому относится item
        merge_w_mean = d[['user_id', 'item_id', 'commodity_desc']].merge(user_feats_with_dept[['user_id', 'commodity_desc', \
                                                                                               'mean_by_commodity', \
                                                                                               'sum_by_commodity', \
                                                                                               'quantity_by_commodity']],
                                                   on=['user_id', 'commodity_desc'], how='left')

        mean_by_item_comm = merge_w_mean.groupby(['user_id', 
                                                  'item_id', 
                                                  'commodity_desc'])['mean_by_commodity', 'sum_by_commodity', \
                                                                     'quantity_by_commodity'].first().reset_index()
        
        mean_by_item_comm.columns = mean_by_item_comm.columns[:3].to_list() \
        + ['mean_in_item_comm', 'sum_in_item_comm', 'quantity_in_item_comm']
        
        d = d.merge(mean_by_item_comm[['user_id', 'item_id', 'mean_in_item_comm', 'sum_in_item_comm', 'quantity_in_item_comm']],
                    on=['user_id', 'item_id'], how='left')
        
        d['mean_in_item_comm_minus_price'] = d.mean_in_item_comm - d.price
        d['mean_in_item_comm_minus_item_mean_for_comm'] = d.mean_in_item_comm - d.item_mean_by_dept
        d['sum_in_item_comm_minus_item_sum_for_comm'] = d.sum_in_item_comm - d.item_sum_by_dept
        d['quantity_in_item_comm_minus_item_quantity_for_comm'] = d.quantity_in_item_comm - d.item_quantity_by_dept

        
        # Средний чек юзера, сумма трат и количество покупок для sub_commodity к которому относится item
        merge_w_mean = d[['user_id', 'item_id', 'sub_commodity_desc']].merge(user_feats_with_dept[['user_id', \
                                                                                                   'sub_commodity_desc', \
                                                                                                   'mean_by_subc', 'sum_by_subc', \
                                                                                                   'quantity_by_subc']],
                                                   on=['user_id', 'sub_commodity_desc'], how='left')

        mean_by_item_subcomm = merge_w_mean.groupby(['user_id', 
                                                  'item_id', 
                                                  'sub_commodity_desc'])['mean_by_subc', 'sum_by_subc', \
                                                                         'quantity_by_subc'].first().reset_index()
        
        mean_by_item_subcomm.columns = mean_by_item_subcomm.columns[:3].to_list() \
        + ['mean_in_item_subc', 'sum_in_item_subc', 'quantity_in_item_subc']
        
        d = d.merge(mean_by_item_subcomm[['user_id', 'item_id', 'mean_in_item_subc', 'sum_in_item_subc', \
                                          'quantity_in_item_subc']], on=['user_id', 'item_id'], how='left')
        
        d['mean_in_item_subc_minus_price'] = d.mean_in_item_subc - d.price
        d['mean_in_item_subc_minus_item_mean_for_subc'] = d.mean_in_item_subc - d.item_mean_by_dept
        d['sum_in_item_subc_minus_item_sum_for_subc'] = d.sum_in_item_subc - d.item_sum_by_dept
        d['quantity_in_item_subc_minus_item_quantity_for_subc'] = d.quantity_in_item_subc - d.item_quantity_by_dept

        
        # Средний чек юзера, сумма трат и количество покупок для brand к которому относится item
        merge_w_mean = d[['user_id', 'item_id', 'brand']].merge(user_feats_with_dept[['user_id', 'brand', 'mean_by_brand', \
                                                                                      'sum_by_brand', 'quantity_by_brand']],
                                                   on=['user_id', 'brand'], how='left')

        mean_by_item_brand = merge_w_mean.groupby(['user_id', 
                                                  'item_id', 
                                                  'brand'])['mean_by_brand', 'sum_by_brand', \
                                                            'quantity_by_brand'].first().reset_index()
        
        mean_by_item_brand.columns = mean_by_item_brand.columns[:3].to_list() \
        + ['mean_in_item_brand', 'sum_in_item_brand', 'quantity_in_item_brand']
        
        d = d.merge(mean_by_item_brand[['user_id', 'item_id', 'mean_in_item_brand', 'sum_in_item_brand', \
                                        'quantity_in_item_brand']], on=['user_id', 'item_id'], how='left')
        
        d['mean_in_item_brand_minus_price'] = d.mean_in_item_brand - d.price

        
        # Средний чек юзера, сумма трат и количество покупок для manufacturer к которому относится item
        merge_w_mean = d[['user_id', 'item_id', 'manufacturer']].merge(user_feats_with_dept[['user_id', 'manufacturer', \
                                                                                             'mean_by_mfd', 'sum_by_mfd', \
                                                                                             'quantity_by_mfd']],
                                                   on=['user_id', 'manufacturer'], how='left')

        mean_by_item_mfd = merge_w_mean.groupby(['user_id', 
                                                  'item_id', 
                                                  'manufacturer'])['mean_by_mfd', 'sum_by_mfd', \
                                                                   'quantity_by_mfd'].first().reset_index()
        
        mean_by_item_mfd.columns = mean_by_item_mfd.columns[:3].to_list() \
        + ['mean_in_item_mfd', 'sum_in_item_mfd', 'quantity_in_item_mfd']
        
        d = d.merge(mean_by_item_mfd[['user_id', 'item_id', 'mean_in_item_mfd', 'sum_in_item_mfd', 'quantity_in_item_mfd']], 
                    on=['user_id', 'item_id'], how='left')
        
        d['mean_in_item_mfd_minus_price'] = d.mean_in_item_mfd - d.price

        
        # Средний чек юзера, сумма трат и количество покупок для размера упаковки item-а
        merge_w_mean = d[['user_id', 'item_id', 'curr_size_of_product']].merge(user_feats_with_dept[['user_id', \
                                                                                                     'curr_size_of_product', \
                                                                                                     'mean_by_size', \
                                                                                                     'sum_by_size', \
                                                                                                     'quantity_by_size']],
                                                   on=['user_id', 'curr_size_of_product'], how='left')

        mean_by_item_size = merge_w_mean.groupby(['user_id', 
                                                  'item_id', 
                                                  'curr_size_of_product'])['mean_by_size', 'sum_by_size', \
                                                                           'quantity_by_size'].first().reset_index()
        
        mean_by_item_size.columns = mean_by_item_size.columns[:3].to_list() \
        + ['mean_in_item_size', 'sum_in_item_size', 'quantity_in_item_size']
        
        d = d.merge(mean_by_item_size[['user_id', 'item_id', 'mean_in_item_size', 'sum_in_item_size', 'quantity_in_item_size']], 
                    on=['user_id', 'item_id'], how='left')
        
        d['mean_in_item_size_minus_price'] = d.mean_in_item_size - d.price

        
        return d
    return fe_lambdas

def get_engineered_features(feat_matrix, user_features, item_features):
    
    ## ЮЗЕРЫ
    
    # Средний чек юзера. Общая сумма покупок / количество покупок
    overall_value = feat_matrix.groupby('user_id')['sales_value'].sum()
    overall_quantity = feat_matrix.groupby('user_id')['quantity'].sum()
    add_user_feats = pd.DataFrame(overall_value/overall_quantity)
    add_user_feats.columns=('mean_value',)
    
    # Сколько недель юзер делал покупки
    weeks = feat_matrix.groupby(['user_id'])['week_no'].nunique().rename('weeks')
    # Среднее количество покупок в месяц (всего покупок / (количество недель / 4))
    add_user_feats = add_user_feats.merge((overall_quantity/(weeks/4)).
                                          rename('mean_quantity_per_month'), on='user_id', how='left')
    
    # Для фичей о среднем чеке пользователей в тех или иных категориях нам нужны категории в матрице интеракций
    feat_matrix_dept = feat_matrix.merge(item_features[['item_id', 'department', 'commodity_desc', 'sub_commodity_desc',\
                                                        'brand', 'manufacturer', 'curr_size_of_product']],\
                                         on='item_id', how='left')
    
    feat_matrix_dept = feat_matrix_dept.merge(weeks, on='user_id', how='left')
    
    # Статистики для юзера по департаменту
    sum_by_dept = feat_matrix_dept.groupby(['user_id', 'department'])['sales_value'].transform('sum')
    quantity_by_dept = feat_matrix_dept.groupby(['user_id', 'department'])['quantity'].transform('sum')
    feat_matrix_dept['mean_by_dept'] = sum_by_dept/quantity_by_dept
    feat_matrix_dept['sum_by_dept'] = sum_by_dept
    feat_matrix_dept['quantity_by_dept'] = quantity_by_dept
    feat_matrix_dept['mean_by_dept_per_week'] = (sum_by_dept/quantity_by_dept)/feat_matrix_dept['weeks']
    feat_matrix_dept['sum_by_dept_per_week'] = sum_by_dept/feat_matrix_dept['weeks']
    feat_matrix_dept['quantity_by_dept_per_week'] = quantity_by_dept/feat_matrix_dept['weeks']
    
    # Статистики для юзера по коммодити, сабкоммодити, brand, производителю и размеру упаковки
    sum_by_comm = feat_matrix_dept.groupby(['user_id', 'commodity_desc'])['sales_value'].transform('sum')
    quantity_by_comm = feat_matrix_dept.groupby(['user_id', 'commodity_desc'])['quantity'].transform('sum')
    feat_matrix_dept['mean_by_commodity'] = sum_by_comm/quantity_by_comm
    feat_matrix_dept['sum_by_commodity'] = sum_by_comm
    feat_matrix_dept['quantity_by_commodity'] = quantity_by_comm

    sum_by_sub = feat_matrix_dept.groupby(['user_id', 'sub_commodity_desc'])['sales_value'].transform('sum')
    quantity_by_sub = feat_matrix_dept.groupby(['user_id', 'sub_commodity_desc'])['quantity'].transform('sum')
    feat_matrix_dept['mean_by_subc'] = sum_by_sub/quantity_by_sub
    feat_matrix_dept['sum_by_subc'] = sum_by_sub
    feat_matrix_dept['quantity_by_subc'] = quantity_by_sub

    sum_by_brand = feat_matrix_dept.groupby(['user_id', 'brand'])['sales_value'].transform('sum')
    quantity_by_brand = feat_matrix_dept.groupby(['user_id', 'brand'])['quantity'].transform('sum')
    feat_matrix_dept['mean_by_brand'] = sum_by_brand/quantity_by_brand
    feat_matrix_dept['sum_by_brand'] = sum_by_brand
    feat_matrix_dept['quantity_by_brand'] = quantity_by_brand

    sum_by_manfd = feat_matrix_dept.groupby(['user_id', 'manufacturer'])['sales_value'].transform('sum')
    quantity_by_manfd = feat_matrix_dept.groupby(['user_id', 'manufacturer'])['quantity'].transform('sum')
    feat_matrix_dept['mean_by_mfd'] = sum_by_manfd/quantity_by_manfd
    feat_matrix_dept['sum_by_mfd'] = sum_by_manfd
    feat_matrix_dept['quantity_by_mfd'] = quantity_by_manfd

    sum_by_size = feat_matrix_dept.groupby(['user_id', 'curr_size_of_product'])['sales_value'].transform('sum')
    quantity_by_size = feat_matrix_dept.groupby(['user_id', 'curr_size_of_product'])['quantity'].transform('sum')
    feat_matrix_dept['mean_by_size'] = sum_by_size/quantity_by_size
    feat_matrix_dept['sum_by_size'] = sum_by_size
    feat_matrix_dept['quantity_by_size'] = quantity_by_size
    
    user_feats_with_dept = add_user_feats.merge(feat_matrix_dept[['user_id', 'department', 'commodity_desc', \
                                                                  'sub_commodity_desc', 'brand', 'manufacturer', \
                                                                  'curr_size_of_product',
                                                              'mean_by_dept', 'sum_by_dept', 'quantity_by_dept',
                                                              'mean_by_dept_per_week', 'sum_by_dept_per_week', \
                                                                  'quantity_by_dept_per_week',
                                                              'mean_by_commodity', 'sum_by_commodity', 'quantity_by_commodity',
                                                              'mean_by_subc', 'sum_by_subc', 'quantity_by_subc',
                                                              'mean_by_brand', 'sum_by_brand', 'quantity_by_brand',
                                                              'mean_by_mfd', 'sum_by_mfd', 'quantity_by_mfd',
                                                              'mean_by_size', 'sum_by_size', 'quantity_by_size']], 
                                            on='user_id', how='left')
    
    mean_by_deps = pd.pivot_table(user_feats_with_dept, 
                  index='user_id', columns='department', 
                  values='mean_by_dept', 
                  aggfunc='first', 
                  fill_value=0
                 )
    
    # Пачка фичей сразу. Мерджим их в юзер-фичи
    add_user_feats = add_user_feats.merge(mean_by_deps, on='user_id', how='left')
    
    # Исходные юзер фичи
    add_user_feats = add_user_feats.merge(user_features, on='user_id', how='left')
    
    
    ## АЙТЕМЫ
    
    # Цена
    add_item_features = pd.DataFrame((feat_matrix.groupby('item_id')['sales_value'].sum()/ 
                                 feat_matrix.groupby('item_id')['quantity'].sum()).rename('price'))
    
    # Среднее количество покупок в неделю
    add_item_features = add_item_features.merge((feat_matrix.groupby('item_id')['quantity'].sum()/
                             feat_matrix.groupby('item_id')['week_no'].nunique()).rename('mean_quantity_per_week'),
                            on='item_id', how='left')
    
    # Средняя сумма по товару
    add_item_features = add_item_features.merge((feat_matrix.groupby('item_id')['sales_value'].mean()).rename('item_mean_value'),
                            on='item_id', how='left')
    
    # Средняя сумма по товару в неделю
    add_item_features = add_item_features.merge((feat_matrix.groupby('item_id')['sales_value'].mean()/
                             feat_matrix.groupby('item_id')['week_no'].nunique()).rename('mean_value_per_week'),
                            on='item_id', how='left')
    
    # Средняя сумма по департаментам
    sum_by_dept = feat_matrix_dept.groupby(['item_id', 'department'])['sales_value'].transform('sum')
    quantity_by_dept = feat_matrix_dept.groupby(['item_id', 'department'])['quantity'].transform('sum')
    feat_matrix_dept['item_mean_by_dept'] = sum_by_dept/quantity_by_dept
    feat_matrix_dept['item_sum_by_dept'] = sum_by_dept
    feat_matrix_dept['item_quantity_by_dept'] = quantity_by_dept
    
    item_feats_with_dept = add_item_features.merge(feat_matrix_dept[['item_id', 'department', 'item_mean_by_dept', \
                                                                     'item_sum_by_dept', 'item_quantity_by_dept']], 
                                            on='item_id', how='left')
    
    add_item_features = add_item_features.merge(
        item_feats_with_dept.groupby(['item_id'])['item_mean_by_dept', \
                'item_sum_by_dept', 'item_quantity_by_dept'].first().reset_index(),
                     on='item_id', how='left')
    
    # Раскладываем средние по департаментам
    mean_by_deps = pd.pivot_table(item_feats_with_dept, 
                      index='item_id', columns='department', 
                      values='item_mean_by_dept', 
                      aggfunc='first', 
                      fill_value=0
                     )
    
    add_item_features = add_item_features.merge(mean_by_deps, on='item_id', how='left')
    
    # И тоже вмердживаем эти новые фичи в таблицу исходных признаков айтемов
    all_item_feats = add_item_features.merge(item_features, on='item_id', how='left')
    


    return {'user_features': add_user_feats, 
            'item_features': all_item_feats, 
            'fe_lambdas': closure(user_feats_with_dept)}
