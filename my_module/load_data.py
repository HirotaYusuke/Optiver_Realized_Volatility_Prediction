import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from tqdm import tqdm
import glob

data_dir = 'features'

def pop_row_stock(df):
    row = df.pop('row_id')
    stock = df.pop('stock_id')
    return row, stock

def pop_row_stock_time(df):
    row = df.pop('row_id')
    stock = df.pop('stock_id')
    time = df.pop('time_id')
    return pd.concat((row, stock, time), axis=1)

def load_row_stock_time(_type):
    row = pd.read_pickle(f'{data_dir}/row_id_{_type}.ftr')
    stock = pd.read_pickle(f'{data_dir}/stock_id_{_type}.ftr')
    time = pd.read_pickle(f'{data_dir}/time_id_{_type}.ftr')
    return pd.concat((row, stock, time), axis=1)

def drop_columns(train, test, excl_columns):
    train = train.drop(excl_columns, axis=1)
    test = test.drop(excl_columns, axis=1)
    return train, test

def load_datasets_sec(sec_list: [List[int]], excl_columns: Optional[List[str]] = None):
    # sec_listに0, 150, 300, 450入れるとそれぞれ追加できる。
    # ext_columnsに抜きたいやつを入れると、除外できる。150,300,450の特徴量を消したいときも0秒の時の名前のままで大丈夫。
    if sec_list is None:
        raise ValueError('sec_listが空です')
    
    if 0 in sec_list:
        sec_list.remove(0)
        train = pd.concat([pd.read_pickle(_path) for _path in tqdm(sorted(glob.glob(f"{data_dir}/*_train.ftr")), desc="reading all train features from 0s") if re.search(r'[^0-9]_train.ftr', _path)], axis=1)
        test = pd.concat([pd.read_pickle(_path) for _path in tqdm(sorted(glob.glob(f"{data_dir}/*_test.ftr")), desc="reading all test features from 0s") if re.search(r'[^0-9]_test.ftr', _path)], axis=1)
        X_train = pd.concat((pop_row_stock_time(train), train), axis=1)
        y_train = X_train.pop('target')
        X_test = pd.concat((pop_row_stock_time(test), test), axis=1)
        
        if excl_columns is not None:
            X_train, X_test = drop_columns(X_train, X_test, excl_columns)
            
    else:
        X_train = load_row_stock_time(_type='train')
        y_train = pd.read_pickle(f'{data_dir}/target_train.ftr')
        X_test = load_row_stock_time(_type='test')
    
    if 150 in sec_list or 300 in sec_list or 450 in sec_list:
        for sec in sec_list:
            if not sec in [150, 300, 450]:
                raise ValueError()
            _X_train = pd.concat([pd.read_pickle(_path) for _path in tqdm(sorted(glob.glob(f"{data_dir}/*_{sec}_train.ftr")), desc=f"reading all train features from {sec}s")], axis=1).reset_index(drop=True)
            _X_test = pd.concat([pd.read_pickle(_path) for _path in tqdm(sorted(glob.glob(f"{data_dir}/*_{sec}_test.ftr")), desc=f"reading all test features from {sec}s")], axis=1).reset_index(drop=True)
            
            if excl_columns is not None:
                _X_train, _X_test = drop_columns(_X_train, _X_test, list(map(lambda x: x + f"_{sec}", excl_columns)))
            
            X_train = pd.concat((X_train, _X_train), axis=1)
            X_test = pd.concat((X_test, _X_test), axis=1)
        
    train = pd.concat((X_train, y_train), axis=1)
    
    return train.reset_index(drop=True), X_test.reset_index(drop=True)

def load_all_datasets(type: str = "train"):
    # 全部
    train = pd.concat([pd.read_pickle(_path) for _path in tqdm(glob.glob(f"{data_dir}/*_{type}.ftr"), desc="reading all train features")], axis=1)
    train_row, train_stock = pop_row_stock(train)
    X_train = pd.concat((train_row, train_stock, train), axis=1)
    y_train = train.pop('target')
    
    test = pd.concat([pd.read_pickle(_path) for _path in tqdm(glob.glob(f"{data_dir}/*_test.ftr"), desc="reading all test features")], axis=1)
    test_row, test_stock = pop_row_stock(test)
    X_test = pd.concat((test_row, test_stock, test), axis=1)
    
    return X_train, y_train, X_test

def load_datasets(feats):
    # 指定したやつだけ
    train = pd.concat([pd.read_pickle(f'{data_dir}/{f}_train.ftr') for f in tqdm(feats, desc="reading train features")], axis=1)
    train_row = pd.read_pickle(f'{data_dir}/row_id_train.ftr')
    train_stock = pd.read_pickle(f'{data_dir}/stock_id_train.ftr')
    X_train = pd.concat((train_row, train_stock, train), axis=1)
    y_train = pd.read_pickle(f'{data_dir}/target_train.ftr')
    
    X_test = pd.concat([pd.read_pickle(f'{data_dir}/{f}_test.ftr') for f in tqdm(feats, desc="reading test features")], axis=1)
    test_row = pd.read_pickle(f'{data_dir}/row_id_train.ftr')
    test_stock = pd.read_pickle(f'{data_dir}/stock_id_train.ftr')
    X_test = pd.concat((test_row, test_stock, train), axis=1)
    
    return X_train, y_train, X_test