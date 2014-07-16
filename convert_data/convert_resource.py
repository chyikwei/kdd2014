import pandas as pd
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from collections import Counter
RESOURCE_FILE = 'raw_data/resources.csv'

TYPE_MAP = {
    'Technology': '1',
    'Supplies': '2',
    'Other': '3',
    'Books': '4',
    'Trips': '5',
    'Visitors': '6',
}


def get_total_price(group):
    total_price = 0.0
    for idx, row in group.iterrows():
        if pd.isnull(row['item_quantity']) or pd.isnull(row['item_unit_price']):
            continue
        cnt = row['item_quantity']
        unit_price = row['item_unit_price']

        total_price += float(cnt) * unit_price
    return total_price


def concat_vendorids(group):
    ids = []
    for idx, row in group.iterrows():
        if pd.isnull(row['vendorid']):
            continue
        v_id = int(row['vendorid'])
        ids.append(str(v_id))

    return "|".join(ids)


def concat_types(group):
    ret = Counter()
    for idx, row in group.iterrows():
        if pd.isnull(row['project_resource_type']):
            continue
        r_type = TYPE_MAP.get(row['project_resource_type'])
        ret[r_type] += 1

    all_type_cnt = [str(ret[k]) for k in sorted(TYPE_MAP.values())]

    return "|".join(all_type_cnt)


def convert_file():
    resources = pd.read_csv(RESOURCE_FILE)
    resources = resources.sort('projectid')

    grouped = resources.groupby('projectid')

    ids = []
    resource_unique_cnts = []
    resource_total_cnts = []
    resource_total_prices = []
    resource_veondorids = []
    resource_types = []
    for name, group in grouped:
        unique_cnt = group.shape[0]
        total_cnt = np.sum(group['item_quantity'].fillna(0))
        total_price = get_total_price(group)
        vendor_ids = concat_vendorids(group)
        types = concat_types(group)

        ids.append(name)
        resource_unique_cnts.append(unique_cnt)
        resource_total_cnts.append(total_cnt)
        resource_total_prices.append(total_price)
        resource_veondorids.append(vendor_ids)
        resource_types.append(types)

    data = {
        'projectid': ids,
        'resource_unique_cnt': resource_unique_cnts,
        'resource_total_cnt': resource_total_cnts,
        'resource_total_price': resource_total_prices,
        'resource_veondorids': resource_veondorids,
        'resource_types': resource_types,
    }

    df = pd.DataFrame(data)
    df.to_csv('resource_by_projectid.csv', index=False)


def convert_num_resources(df):
    fields = ['resource_unique_cnt', 'resource_total_cnt', 'resource_total_price']

    data_list = []
    for field in fields:
        data_list.append(df[field].values)

    ret = np.vstack(data_list).T
    ret = sp.csr_matrix(ret)
    return ret


def convert_category_resources(df):
    def transform(text):
        cells = text.split('|')
        cells = [int(c) for c in cells]
        # to binary
        cells = [1.0 if c > 0 else 0.0 for c in cells]
        return cells
    field = 'resource_types'
    data = df[field].values
    data_list = [transform(d) for d in data]

    ret = np.vstack(data_list)
    ret = sp.csr_matrix(ret)
    return ret


def merge_projects_resources():
    projects = pd.read_csv('raw_data/projects.csv')
    projects = projects.sort('projectid')
    resources = pd.read_csv('resource_by_projectid.csv')
    resources = resources.sort('projectid')

    projects_resources = pd.merge(projects, resources, on='projectid', how='left')

    train = projects_resources[projects_resources['date_posted'] < '2014-01-01']
    train_2013 = train[train['date_posted'] >= '2013-01-01']
    train_2013 = train_2013.sort('projectid')
    
    test = projects_resources[projects_resources['date_posted'] >= '2014-01-01']
    test = test.sort('projectid')

    # convert numeric data to matrix
    train_num_mtx = convert_num_resources(train_2013)
    sio.mmwrite('resource_train_2013_num_mtx', train_num_mtx)
    test_num_mtx = convert_num_resources(test)
    sio.mmwrite('resource_test_num_mtx', test_num_mtx)

    # convert resource type category
    train_cat_mtx = convert_category_resources(train_2013)
    sio.mmwrite('resource_train_2013_type_mtx', train_cat_mtx)
    test_cat_mtx = convert_category_resources(test)
    sio.mmwrite('resource_test_type_mtx', test_cat_mtx)

