import numpy as np
import pandas as pd
# import datetime


def transform_cols (df, dict_col_types = None):
    # Расширяйте для необходимых столбцов и их явной типизации
    if dict_col_types is None:
        dict_col_types = {
        'amount_original':(float, 0.0),
        'cdf_s_126':(str, u'null'),
        'cdf_s_138':(str, u'null'),
        'channel_indicator_desc':(str, u'null'),
        'event_description':(str, u'null'),
        'cdf_s_294':(int, 0),
        'cdf_s_140':(float, 0.0),
        'data_i_120':(int, 0),
        'cdf_s_218':(str, u'null'),
        'data_s_65':(int, 0),
        'cdf_s_127':(int, 30),
        'cdf_s_135':(int, 30),
        'cdf_s_130':(int, 30),
        'cdf_s_129':(int, 30),
        'cdf_s_134':(int, 30),
        'data_i_154':(float, -150),
        'cdf_s_133':(int, 30),
        'cdf_s_20':(str, u'null'),
        'cdf_s_299':(str, u'null'),
        'short_date':(int, 0)
        }
                
    if df.shape[0] > 0:
        df.replace(u'null', np.nan, inplace=True)

        for i in dict_col_types:
            if i in df.columns:
                df[i] = df[i].fillna(dict_col_types[i][1]).astype(dict_col_types[i][0])
    
    return df

def calc_base_features(data):
    feat_matrix = pd.DataFrame()
    #data = data[data.event_description.isin([u'Перевод частному лицу',u'Оплата услуг',u'Перевод между своими счетами и картами'])]
    
    if data.shape[0] == 0:
        return feat_matrix
    
    # заполняем ряд пропусков
    data.cdf_s_140 = data.cdf_s_140.fillna(0).astype(float)/1000 # кумулятивная сумма опреаций за сутки, если не заполнена, то значит это первая операций, т.е. = 0
    data.data_i_120.fillna(1, inplace=True)

    
    feat_matrix['event_id'] = data.event_id
    feat_matrix['user_id'] = data.user_id

    feat_matrix['custom_mark'] = data.custom_mark    
    feat_matrix['event_time'] = data.event_time

    feat_matrix['amount'] = data.amount_original
    
    feat_matrix['client_age'] = [x.days/360 for x in (data.event_time - data.cdf_s_19)]
  
        
    feat_matrix['cat_new_ip'] = [1 if x == u'ДА' else 0 if x == u'НЕТ' else 2 for x in data.cdf_s_126]
    feat_matrix['cat_new_prov'] =  [1 if x == u'ДА' else 0 if x == u'НЕТ' else 2 for x in data.cdf_s_138]
    feat_matrix['channel_op'] =  [0 if x == u'MOBILE' else 1 if x == u'WEB' else 2 for x in data.channel_indicator_desc]
    feat_matrix['op_type'] = [0 if x == u'Перевод частному лицу' else 1 if x==u'Оплата услуг' else 2 if x ==u'Перевод между своими счетами и картами' else 3 for x in data.event_description]


    feat_matrix ['recip_age'] =  [1 if x == 0 else 0 for x in data.cdf_s_294] # бинарный флаг определяющий наличие возраста получателя (полезен для линейных моделей, менее для деревьев с учетом следующего признака)
    
    feat_matrix['age_diff'] = feat_matrix.client_age - [int(x) if x != 0 else 1000 for x in data.cdf_s_294] # разница возорастов получателей и отправителей, если отсутствует/неприменимо, то padding 500    
  
    
    feat_matrix['cumulative_sum_total'] = data.cdf_s_140 # кумулятивная сумма операций за сутки в каналах web и МП
    
    feat_matrix['data_i_120'] = data.data_i_120 
    
    
    feat_matrix['relative'] = [1 if x == u'ДА' else 0 for x in data.cdf_s_218] # перевод родственнику
    
    feat_matrix['know_recip_power'] = [ x if x is not None else 0 for x in data.data_s_65] # сила связи отправителя и получателя
    

    feat_matrix['cdf_s_127'] = data.cdf_s_127#.apply(lambda x: 1 if x is not None else 0)
    feat_matrix['cdf_s_135'] = data.cdf_s_135#.apply(lambda x: 1 if x is not None else 0)
    feat_matrix['cdf_s_130'] = data.cdf_s_130#.apply(lambda x: 1 if x is not None else 0)
    feat_matrix['cdf_s_129'] = data.cdf_s_129#.apply(lambda x: 1 if x is not None else 0)
    feat_matrix['cdf_s_134'] = data.cdf_s_134#.apply(lambda x: 1 if x is not None else 0)
    feat_matrix['data_i_154'] = [ x if x is not None else -150 for x in data.data_i_154]
    feat_matrix['cdf_s_133'] = data.cdf_s_133#.apply(lambda x: 1 if x is not None else 0)
    feat_matrix['data_i_120'] = data.data_i_120
    feat_matrix['know_recip_card_age'] = [1 if x is not None else 0 for x in data.cdf_s_124]
    
    
    feat_matrix['recip_card_age'] = [x.days if type(x) is not pd.tslib.NaTType else 912321 for x in (data.event_time - data.cdf_s_124)]
    
    # feat_matrix['cat_client_region'] = [x if x.isdigit() else 912321 for x in data.cdf_s_20]
    feat_matrix['one_region'] = (data.cdf_s_20 == data.cdf_s_299).astype(int) # сравнение регионов
    

    #ADD NEW FEATURES
    feat_matrix['krp_pow2'] = (feat_matrix['know_recip_power']) ** 2
    feat_matrix['log_amount'] = np.log(feat_matrix['amount'] + 1)
    feat_matrix['ip_isp'] = np.array([x if x.isdigit() else 912321 for x in data.cdf_s_20], dtype=float)
    feat_matrix['amnt2chnls'] = (data["amount_original"].fillna(0).astype(float) / \
        (data["cdf_s_136"].fillna(0).astype(float) + data["amount_original"].fillna(0).astype(float) + \
            data["amount_original"].fillna(0) + 1))
    return feat_matrix


def load_data(chunk_names, fields=None, query=None, sample='train', dict_col_types=None):
    df = pd.DataFrame({})
    
    if not isinstance(chunk_names, list):
        chunk_names = [chunk_names]
        
    for chunk_name in chunk_names:
        chunk_df = pd.read_feather(
            '../data/raw_splits/{smpl}/{ch_nm}'.format(smpl=sample, ch_nm=chunk_name))
            
        if fields is None:
            fields = chunk_df.columns.tolist()
            
        if query is None:
            df = pd.concat([df,
                            transform_cols(
                                chunk_df[fields])], ignore_index=True)
        else:
            df = pd.concat([df,
                            transform_cols(
                                chunk_df).query(query)[fields]], ignore_index=True)
    return df


def features_handler(chunk_names, calc_feat, query=None, chunk_size=5000):
    res_df = pd.DataFrame()
    for chunk_name in chunk_names:
        
        feat_chunk = calc_feat(
            load_data(
                chunk_name,
                query=query,
                dict_col_types=None)
        )

        res_df = pd.concat([res_df, feat_chunk], ignore_index=True)
    return res_df


def cust_mark_to_class(custom_mark):
    """
    Преобразует входящее значение CUSTOM_MARK в класс
    return:
        1 - фрод
        0 - легитимная
        -1 - неизвестно
    """
    ret = -1
    if custom_mark in ['F','S']:
        ret = 1
    elif custom_mark in ['A','G', np.NaN]:
        ret = 0
    
    return ret