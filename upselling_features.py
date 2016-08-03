import datetime
from collections import OrderedDict

import pandas
import numpy as np
from tqdm import tqdm



def fix_geo_coordinates(table, prefix=''):
    budapest_x = 650947
    budapest_y = 239670
    geo_scale = 250000
    table['%sGEO_X' % prefix] = table.eval('(%sGEO_X - @budapest_x) / @geo_scale' % prefix)
    table['%sGEO_Y' % prefix] = table.eval('(%sGEO_Y - @budapest_y) / @geo_scale' % prefix)

    
def load_events_table(filename, table_bank_info):
    table_events = pandas.read_csv(filename, parse_dates=['DATE'])\
        .sort_values(['USER_ID', 'POI_ID']).reset_index(drop=True)

    for col in ('TIME_CAT', 'LOC_CAT', 'MC_CAT', 'CARD_CAT', 'AMT_CAT'):
        table_events[col] = table_events[col].astype('category')
        
    fix_float = lambda value: float(value) if value != '-' else None
    table_events['GEO_X'] = table_events['GEO_X'].map(fix_float)
    table_events['GEO_Y'] = table_events['GEO_Y'].map(fix_float)

    table_events = table_events.join(table_bank_info, on='POI_ID', rsuffix='_')
    sel = table_events['GEO_X'].isnull() & table_events['GEO_X_'].notnull()
    table_events.ix[sel, 'GEO_X'] = table_events.ix[sel, 'GEO_X_']
    sel = table_events['GEO_Y'].isnull() & table_events['GEO_Y_'].notnull()
    table_events.ix[sel, 'GEO_Y'] = table_events.ix[sel, 'GEO_Y_']

    fix_geo_coordinates(table_events)
    
    table_events.drop(['GEO_X_', 'GEO_Y_'], axis=1, inplace=True)
    
    return table_events


def load_users_table(filename):
    table_users = pandas.read_csv(filename, index_col='USER_ID').sort_index()

    fix_geo_coordinates(table_users, prefix='LOC_')

    table_users['GEN'] = table_users['GEN'].map(lambda g: 'm' if g == 1 else 'f')

    for col in ('AGE_CAT', 'LOC_CAT', 'INC_CAT', 'GEN'):
        table_users[col] = table_users[col].astype('category')

    table_users['CARD_MONTHLY'] = ''
    table_users['WEALTH_MONTHLY'] = ''

    for col in table_users.columns:
        if col.startswith('C201'):
            table_users['CARD_MONTHLY'] += table_users[col].astype(str)
        if col.startswith('W201'):
            table_users['WEALTH_MONTHLY'] += table_users[col].astype(str)

    table_users.drop([ 
        col for col in table_users.columns 
        if col.startswith('C201') or col.startswith('W201')
    ], axis=1, inplace=True)

    import datetime
    if 'TARGET_TASK_2' in table_users.columns:
        table_users['TARGET_TASK_2'] = pandas.to_datetime(table_users['TARGET_TASK_2'].map(
            lambda dt: datetime.datetime.strptime(dt, '%Y.%m.%d') if dt != '-' else None))
    
    return table_users

def encode_ordered_cat(key, values=['a', 'b', 'c', 'd', 'e', 'f']):
    for i, letter in enumerate(values):
        if key == letter:
            return i+1
    return 0

def encode_one_hot(prefix, key, values=['a', 'b', 'c', 'd', 'e', 'f']):
    return [
        (prefix + '_' + value, int(value == key))
        for value in values
    ]

def encode_hist(prefix, events_value, values=None):
    if values is None:
        values = events_value.cat.categories
    n = events_value.shape[0]
    return [
        (prefix + '_' + value, (events_value == value).sum())
        for value in values
    ] + [
        (prefix + '_r_' + value, (float((events_value == value).sum()) / n) if n > 0 else 0)
        for value in values
    ]

def extract_features(table_users, table_events, train_target=False, year=2014):
    table_events_groups = table_events.groupby('USER_ID')
    
    for user_id, user_info in tqdm(table_users.iterrows(), total=table_users.shape[0]):
        event_indices = table_events_groups.groups.get(user_id) or np.array([])
        user_events = table_events.ix[event_indices, :]

        features = [
            ('user_id', user_id),
        ]
        
        if train_target:
            features += [
                ('apply_already', int(user_info.TARGET_TASK_2 < pandas.to_datetime('2014-07-01'))),
                ('apply_target', int((user_info.TARGET_TASK_2 > pandas.to_datetime('2014-06-30')) &
                     (user_info.TARGET_TASK_2 < pandas.to_datetime('2015-01-01')))),
                ('apply_future', int((user_info.TARGET_TASK_2 >= pandas.to_datetime('2014-07-01')))),
            ]

        # categorical features
        features += [
            ('age_cat', encode_ordered_cat(user_info.AGE_CAT)),
            ('loc_cat', encode_ordered_cat(user_info.LOC_CAT)),
            ('inc_cat', encode_ordered_cat(user_info.INC_CAT, ['d', 'a', 'b', 'c'])),
        ]
        features += encode_one_hot('age_cat', user_info.AGE_CAT, ['a', 'b', 'c'])
        features += encode_one_hot('loc_cat', user_info.LOC_CAT, ['a', 'b', 'c'])
        features += encode_one_hot('inc_cat', user_info.INC_CAT, ['a', 'b', 'c', 'd'])
        features += encode_one_hot('gender', user_info.GEN, ['m', 'f'])
        
        # geo features
        features += [
            ('loc_x', user_info.LOC_GEO_X),
            ('loc_y', user_info.LOC_GEO_Y),
        ]
        
        event_loc_dist = np.sqrt(((user_events.GEO_X - user_info.LOC_GEO_X)**2 + 
                          (user_events.GEO_Y - user_info.LOC_GEO_Y)**2))
        event_loc_angle = np.arctan2(
            (user_events.GEO_Y - user_info.LOC_GEO_X), 
            (user_events.GEO_X - user_info.LOC_GEO_X),
        )
        
        event_center_dist = np.sqrt(user_events.GEO_X**2 + user_events.GEO_Y**2)
        event_center_angle = np.arctan2(user_events.GEO_Y, user_events.GEO_X)
        
        features += [
            ('event_loc_x_med', user_events.GEO_X.median()),
            ('event_loc_y_med', user_events.GEO_Y.median()),
            ('event_loc_x_med', (user_events.GEO_X - user_info.LOC_GEO_X).mean()),
            ('event_loc_y_med', (user_events.GEO_Y - user_info.LOC_GEO_Y).mean()),
            ('event_loc_dist_min', event_loc_dist.min()), 
            ('event_loc_dist_max', event_loc_dist.max()), 
            ('event_loc_dist_mean', event_loc_dist.mean()), 
            ('event_loc_dist_std', event_loc_dist.std()), 
            ('event_loc_dist_q2', event_loc_dist.quantile(0.2)),
            ('event_loc_dist_q5', event_loc_dist.quantile(0.5)),
            ('event_loc_dist_q8', event_loc_dist.quantile(0.8)),
            ('event_loc_angle_mean', event_loc_angle.mean()),
            ('event_loc_angle_std', event_loc_angle.std()),
            ('event_loc_angle_min', event_loc_angle.min()),
            ('event_loc_angle_max', event_loc_angle.max()),
            ('event_loc_angle_diff', event_loc_angle.max() - event_loc_angle.min()),
            ('event_loc_angle_med', event_loc_angle.median()),
            ('event_center_dist_min', event_center_dist.min()), 
            ('event_center_dist_max', event_center_dist.max()), 
            ('event_center_dist_mean', event_center_dist.mean()), 
            ('event_center_dist_std', event_center_dist.std()), 
            ('event_center_dist_q2', event_center_dist.quantile(0.2)),
            ('event_center_dist_q5', event_center_dist.quantile(0.5)),
            ('event_center_dist_q8', event_center_dist.quantile(0.8)),
            ('event_center_angle_mean', event_center_angle.mean()),
            ('event_center_angle_std', event_center_angle.std()),
            ('event_center_angle_min', event_center_angle.min()),
            ('event_center_angle_max', event_center_angle.max()),
            ('event_center_angle_diff', event_center_angle.max() - event_center_angle.min()),
            ('event_center_angle_med', event_center_angle.median()),
        ]

        # card & wealth        
        card_monthly = map(int, user_info.CARD_MONTHLY[:6])
        wealth_monthly = map(int, user_info.WEALTH_MONTHLY[:6])
        features += [
            ('card_months', sum(card_monthly)),
            ('card_last', card_monthly[-1]),
            ('card_inc', sum(map(lambda (a, b): int(a>b), zip(card_monthly[:-1], card_monthly[1:])))),
            ('card_dec', sum(map(lambda (a, b): int(a<b), zip(card_monthly[:-1], card_monthly[1:])))),
            ('wealth_months', sum(wealth_monthly)),
            ('wealth_last', wealth_monthly[-1]),
            ('wealth_inc', sum(map(lambda (a, b): int(a>b), zip(wealth_monthly[:-1], wealth_monthly[1:])))),
            ('wealth_dec', sum(map(lambda (a, b): int(a<b), zip(wealth_monthly[:-1], wealth_monthly[1:])))),
        ]
        
        # events
        n_events = user_events.shape[0]
        features += [
            ('events', n_events),
            
            ('uniq_poi', user_events.POI_ID.nunique()),
            ('uniq_poi_b', user_events.query('CHANNEL == "b"').POI_ID.nunique()),
            ('uniq_poi_n', user_events.query('CHANNEL == "n"').POI_ID.nunique()),
            ('uniq_poi_p', user_events.query('CHANNEL == "p"').POI_ID.nunique()),
        ]
        features += encode_hist('event_channel', user_events.CHANNEL, values=['b', 'n', 'p'])
        features += encode_hist('event_time', user_events.TIME_CAT)
        features += encode_hist('event_loc', user_events.LOC_CAT)
        features += encode_hist('event_mc', user_events.MC_CAT)
        features += encode_hist('event_card', user_events.CARD_CAT)
        features += encode_hist('event_amt', user_events.AMT_CAT)
        
        # weekday
        features += [
            ('events_weekday_%d' % weekday, (user_events.DATE.dt.dayofweek == weekday).sum())
            for weekday in xrange(7)
        ]
        features += [
            ('events_weekday_r_%d' % weekday, (user_events.DATE.dt.dayofweek == weekday).mean())
            for weekday in xrange(7)
        ]
        
        # activity period
        
        active_days = user_events.DATE.dt.date.nunique()
        active_weeks = user_events.DATE.dt.weekofyear.nunique()
        active_range_days = (user_events.DATE.dt.date.max() - user_events.DATE.dt.date.min()).days if n_events > 0 else np.nan
        idle_days = (datetime.date(year, 7, 1) - user_events.DATE.dt.date.max()).days if n_events > 0 else np.nan
        features += [
            ('active_days', active_days),
            ('active_weeks', active_weeks),
            ('active_range_days', active_range_days),
            ('active_days_density', (float(active_days) / active_range_days) if active_range_days > 0 else np.nan),
            ('active_weeks_density', (float(active_weeks) / (float(active_range_days) / 7)) if active_range_days > 0 else np.nan),
            ('idle_days', idle_days),
            ('idle_days_rate', (float(idle_days) / (idle_days + active_range_days)) if idle_days + active_range_days > 0 else np.nan),
            ('active_days_rate', (float(active_range_days) / (idle_days + active_range_days)) if idle_days + active_range_days > 0 else np.nan),
        ]

        yield OrderedDict(features)


def create_dataframe(row_generator):
    def generator2():
        for row in row_generator:
            yield row
            if generator2.columns is None:
                generator2.columns = row.keys()
    generator2.columns = None
    return pandas.DataFrame.from_records(generator2()).ix[:, generator2.columns]


def create_feature_tables():
    print ('Loading data tables')
    
    table_bank_info = pandas.read_csv('data/bank_info.csv')
    table_bank_info['POI_ID'] = table_bank_info['POI_ID'].astype(int)
    table_bank_info.set_index('POI_ID', inplace=True)
    
    table_users_2014 = load_users_table('data/users_2014.csv')
    table_users_2015 = load_users_table('data/users_2015.csv')
    table_events_2014 = load_events_table('data/train_2014.csv', table_bank_info)
    table_events_2015 = load_events_table('data/train_2015.csv', table_bank_info)
    
    print ('Extracting train features')
    features_train = create_dataframe(
        extract_features(
            table_users_2014,
            table_events_2014.ix[table_events_2014.DATE < pandas.to_datetime('2014-07-01'), :],
            train_target=True,
            year=2014,
        )
    )
    features_train.to_csv('features/upselling_train.csv', index=False)

    print ('Extracting test features')
    features_test = create_dataframe(
        extract_features(
            table_users_2015,
            table_events_2015,
            year=2015,
        )
    )
    features_test.to_csv('features/upselling_test.csv', index=False)


if __name__ == '__main__':
    create_feature_tables()
    