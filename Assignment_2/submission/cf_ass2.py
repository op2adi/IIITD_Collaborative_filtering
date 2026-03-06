#!/usr/bin/env python3

import os,time,json,math,warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import RandomizedSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import joblib
import logging
from pathlib import Path
import optuna
from scipy import stats
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

np.random.seed(42)


dt_pth = "archive/ml-100k"   
ltdim = 60                 # I thought to change but time constraints 
thrds = min(8,os.cpu_count() or 4)
rs = 42
flds = [1,2,3,4,5]
rscsv = "results.csv"
rpt = "results"
sv_dir = "saved_models"

tuner = True        
mthd = "optuna"        
trls = 8               
cross_v = 2
naukri_jbs = 1           


cf_Chk = True
siamese_chk = True
fft_mlp = True        


# logging
LOG_FILE = "run_no_meta.log"

# -------------------------
os.makedirs(rpt,exist_ok=True)
os.makedirs(sv_dir,exist_ok=True)
os.makedirs("fold_cache1",exist_ok=True)



# logging
logger = logging.getLogger("ml100k_no_meta")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)
fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(fmt)
logger.addHandler(fh)
logger.info(f"thrds={thrds}")

# optional libs
try:
    from xgboost import XGBClassifier
    HAS_XGB = True; logger.info("XGBoost available")
except Exception:
    HAS_XGB = False; logger.info("XGBoost not available")
try:
    import lightgbm as lgb
    HAS_LGB = True; logger.info("LightGBM available")
except Exception:
    HAS_LGB = False; logger.info("LightGBM not available")
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True; logger.info("CatBoost available")
except Exception:
    HAS_CAT = False; logger.info("CatBoost not available")
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True; logger.info("TensorFlow available")
except Exception:
    HAS_TF = False; logger.info("TensorFlow not available")

def fit_baseline_biases(train_df,reg=10.0,n_iter=10):
    mu = float(train_df['rating'].mean())
    users_arr = np.sort(train_df['user_id'].unique())
    items_arr = np.sort(train_df['item_id'].unique())
    user_idx = {u:i for i,u in enumerate(users_arr)}
    item_idx = {i:j for j,i in enumerate(items_arr)}
    n_u,n_i = len(users_arr),len(items_arr)
    bu = np.zeros(n_u); bi = np.zeros(n_i)
    counts_u = np.zeros(n_u); counts_i = np.zeros(n_i)
    U_to_items = defaultdict(list)
    I_to_users = defaultdict(list)
    for _,r in train_df.iterrows():
        u = user_idx[r['user_id']]; i = item_idx[r['item_id']]
        U_to_items[u].append((i,r['rating'])); I_to_users[i].append((u,r['rating']))
        counts_u[u] += 1; counts_i[i] += 1
    for _ in range(n_iter):
        for i,recs in I_to_users.items():
            numer = sum((r - mu - bu[u]) for (u,r) in recs)
            bi[i] = numer / (counts_i[i] + reg)
        for u,recs in U_to_items.items():
            numer = sum((r - mu - bi[i]) for (i,r) in recs)
            bu[u] = numer / (counts_u[u] + reg)
    bu_map = {users_arr[i]: bu[i] for i in range(n_u)}
    bi_map = {items_arr[i]: bi[i] for i in range(n_i)}
    return mu,bu_map,bi_map

def compute_svd_fast(train_df,latent_dim,max_user_id,max_item_id):
    R = csr_matrix((train_df['rating'].values.astype('float32'),
                    (train_df['user_id'].values - 1,train_df['item_id'].values - 1)),
                   shape=(max_user_id,max_item_id),dtype=np.float32)
    svd = TruncatedSVD(n_components=latent_dim,random_state=rs,n_iter=7)
    user_lat = svd.fit_transform(R)
    item_lat = svd.components_.T
    return user_lat.astype('float32'),item_lat.astype('float32')

# caching helper
def cache_fold_paths(folder,fold):
    Path(folder).mkdir(parents=True,exist_ok=True)
    return {
        'X_train_path': os.path.join(folder,f"X_train_fold{fold}.joblib"),
        'X_test_path':  os.path.join(folder,f"X_test_fold{fold}.joblib"),
        'y_train_path': os.path.join(folder,f"y_train_fold{fold}.npy"),
        'y_test_path':  os.path.join(folder,f"y_test_fold{fold}.npy"),
        'meta_path':    os.path.join(folder,f"meta_fold{fold}.joblib"),
        'X_train_nb_path': os.path.join(folder,f"X_train_nb_fold{fold}.joblib"),
        'X_test_nb_path':  os.path.join(folder,f"X_test_nb_fold{fold}.joblib"),
    }

def build_features_train_test_cached(train_df, test_df, latent_dim, fold, cache_dir="fold_cache"):
    paths = cache_fold_paths(cache_dir, fold)

    required = [
        "X_train_path", "X_test_path",
        "y_train_path", "y_test_path",
        "meta_path",
        "X_train_nb_path", "X_test_nb_path"
    ]

    if all(os.path.exists(paths[k]) for k in required):
        X_train = joblib.load(paths["X_train_path"])
        X_test = joblib.load(paths["X_test_path"])
        y_train = np.load(paths["y_train_path"])
        y_test = np.load(paths["y_test_path"])
        X_train_nb = joblib.load(paths["X_train_nb_path"])
        X_test_nb = joblib.load(paths["X_test_nb_path"])
        meta = joblib.load(paths["meta_path"])
        return X_train, X_test, y_train, y_test, X_train_nb, X_test_nb, meta

    mu, bu_map, bi_map = fit_baseline_biases(train_df, reg=10.0, n_iter=10)

    train = train_df.copy()
    test = test_df.copy()

    train = train.assign(
        mu=mu,
        bu=train["user_id"].map(bu_map).fillna(0.0),
        bi=train["item_id"].map(bi_map).fillna(0.0),
    )

    test = test.assign(
        mu=mu,
        bu=test["user_id"].map(bu_map).fillna(0.0),
        bi=test["item_id"].map(bi_map).fillna(0.0),
    )

    user_stats = train.groupby("user_id")["rating"].agg(["mean", "count"])
    user_stats.columns = ["u_mean", "u_count"]

    item_stats = train.groupby("item_id")["rating"].agg(["mean", "count"])
    item_stats.columns = ["i_mean", "i_count"]

    train = train.join(user_stats, on="user_id").join(item_stats, on="item_id")
    test = test.join(user_stats, on="user_id").join(item_stats, on="item_id")

    for col in ["u_mean", "u_count", "i_mean", "i_count"]:
        train[col] = train[col].fillna(0.0)
        test[col] = test[col].fillna(0.0)

    max_user_id = int(max(train_df["user_id"].max(), test_df["user_id"].max()))
    max_item_id = int(max(train_df["item_id"].max(), test_df["item_id"].max()))

    user_lat, item_lat = compute_svd_fast(train_df, latent_dim, max_user_id, max_item_id)

    for k in range(latent_dim):
        train[f"u_lat_{k}"] = user_lat[train["user_id"].values - 1, k]
        train[f"i_lat_{k}"] = item_lat[train["item_id"].values - 1, k]
        test[f"u_lat_{k}"] = user_lat[test["user_id"].values - 1, k]
        test[f"i_lat_{k}"] = item_lat[test["item_id"].values - 1, k]

    num_cols = (
        ["mu", "bu", "bi", "u_mean", "u_count", "i_mean", "i_count"]
        + [f"u_lat_{k}" for k in range(latent_dim)]
        + [f"i_lat_{k}" for k in range(latent_dim)]
    )

    train[num_cols] = train[num_cols].fillna(0.0)
    test[num_cols] = test[num_cols].fillna(0.0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[num_cols].values.astype("float32"))
    X_test = scaler.transform(test[num_cols].values.astype("float32"))

    y_train = train["rating"].values.astype("int32")
    y_test = test["rating"].values.astype("int32")

    raw_train = train[num_cols].values.astype("float32")
    min_val = np.nanmin(raw_train)

    X_train_nb = raw_train - min_val
    X_test_nb = test[num_cols].values.astype("float32") - min_val

    meta = {
        "scaler": scaler,
        "mu": mu,
        "bu_map": bu_map,
        "bi_map": bi_map,
        "latent_dim": latent_dim,
        "max_user_id": max_user_id,
        "max_item_id": max_item_id,
        "num_cols": num_cols,
    }

    joblib.dump(X_train, paths["X_train_path"])
    joblib.dump(X_test, paths["X_test_path"])
    joblib.dump(X_train_nb, paths["X_train_nb_path"])
    joblib.dump(X_test_nb, paths["X_test_nb_path"])
    np.save(paths["y_train_path"], y_train)
    np.save(paths["y_test_path"], y_test)
    joblib.dump(meta, paths["meta_path"])

    return X_train, X_test, y_train, y_test, X_train_nb, X_test_nb, meta

def make_model_zoo():
    # model maker for the code 
    zoo = {}
    zoo['Logistic'] = LogisticRegression(max_iter=1500,multi_class='multinomial',solver='saga',n_jobs=thrds)
    zoo['SGD_Log'] = SGDClassifier(loss='log_loss',max_iter=2000)
    zoo['ComplementNB'] = ComplementNB()
    zoo['KNN'] = KNeighborsClassifier(n_jobs=thrds)
    zoo['LinearSVC'] = LinearSVC(max_iter=5000)
    zoo['RandomForest'] = RandomForestClassifier(n_estimators=200,class_weight='balanced',n_jobs=thrds,random_state=rs)
    zoo['ExtraTrees'] = ExtraTreesClassifier(n_estimators=150,n_jobs=thrds,random_state=rs)
    zoo['MLP'] = MLPClassifier(hidden_layer_sizes=(512,256),max_iter=200,random_state=rs)
    if HAS_XGB:
        xgb_params = {'use_label_encoder':False,'eval_metric':'mlogloss','n_estimators':150,'random_state':rs}
        xgb_params.update({'tree_method':'gpu_hist','predictor':'gpu_predictor','gpu_id':0} if 'CUDA_VISIBLE_DEVICES' in os.environ else {})
        zoo['XGBoost'] = XGBClassifier(**xgb_params)
    if HAS_LGB:
        lgb_params = {'n_estimators':150,'random_state':rs}
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            lgb_params.update({'device':'gpu'})
        zoo['LightGBM'] = lgb.LGBMClassifier(**lgb_params)
    if HAS_CAT:
        cat_params = {'iterations':150,'verbose':0,'random_seed':rs}
        zoo['CatBoost'] = CatBoostClassifier(**cat_params)
    return zoo
# used code from optuna directly 
def tune_model(model_name,estimator,X,y,n_trials=trls,cv=cross_v,n_jobs=naukri_jbs):
    logger.info(f"Starting tuning for {model_name} (trials={n_trials},cv={cv})")
    results_path = os.path.join(sv_dir,f"tuning_results_{model_name}.csv")
    best_params_path = os.path.join(sv_dir,f"best_params_{model_name}.json")

    # define small safe search grids
    if model_name == 'RandomForest':
        param_dist = {
            'n_estimators': [100,150,200],
            'max_depth': [None,10,25],
            'max_features': ['sqrt',0.6],
            'min_samples_split': [2,4]
        }
    elif model_name == 'LightGBM' and HAS_LGB:
        param_dist = {
            'num_leaves': [31,50,80],
            'n_estimators': [100,150,200],
            'max_depth': [-1,8,12],
            'learning_rate': [0.05,0.1]
        }
    elif model_name == 'XGBoost' and HAS_XGB:
        param_dist = {
            'max_depth': [3,6],
            'n_estimators': [100,150,200],
            'learning_rate': [0.05,0.1],
            'subsample': [0.7,1.0]
        }
    elif model_name == 'CatBoost' and HAS_CAT:
        param_dist = {
            'depth': [4,6,8],
            'iterations': [100,150,200],
            'learning_rate': [0.05,0.1]
        }
    # elif model_name == 'HistGB':
    #     param_dist = {
    #         'max_iter': [50,80,120],
    #         'learning_rate': [0.05,0.1],
    #         'max_depth': [8,12]
    #     }
    else:
        param_dist = {
            'C': [1e-3,1e-2,1e-1,1,10,100] if model_name in ['Logistic','LinearSVC','SGD_Log'] else []
        }

    # If optuna
    # debug kr rha tha 
    # print(X.shape)
    use_optuna = (mthd == 'optuna')
    if use_optuna:
        def optuna_obj(trial):
            params = {}
            for p,vals in param_dist.items():
                if isinstance(vals,list) and len(vals) > 0:
                    params[p] = trial.suggest_categorical(p,vals)
                else:
                    params[p] = trial.suggest_float(p,1e-5,1.0)
            est = clone(estimator)
            try:
                est.set_params(**params)
            except Exception:
                pass
           
            t0 = time.time()
            scores = cross_val_score(est,X,y,cv=cv,scoring='accuracy',n_jobs=1)
            dur = time.time() - t0
            logger.info(f"[optuna:{model_name}] trial {trial.number} mean_score={scores.mean():.4f} time={dur:.1f}s params={params}")
            return float(scores.mean())
        study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=rs)) # maximize kr rha 
        study.optimize(optuna_obj,n_trials=n_trials,n_jobs=1)
        best_params = study.best_params
        best_score = study.best_value
        logger.info(f"Optuna best for {model_name}: score={best_score:.4f} params={best_params}")
        with open(best_params_path,'w') as fh:
            json.dump({'best_score': float(best_score),'best_params': best_params},fh,indent=2)
        return best_params


    rd = {p: v for p,v in param_dist.items() if v}
    if not rd:
        logger.info(f"No tuning space for {model_name}; skipping tuning.")
        return {}
    rs = RandomizedSearchCV(estimator,param_distributions=rd,n_iter=n_trials,scoring='accuracy',cv=cv,random_state=rs,n_jobs=n_jobs,verbose=0)
    rs.fit(X,y)
    best_params = rs.best_params_
    best_score = rs.best_score_
    logger.info(f"RandomSearch best for {model_name}: score={best_score:.4f} params={best_params}")
    try:
        df_cv = pd.DataFrame(rs.cv_results_)
        df_cv.to_csv(results_path,index=False)
    except Exception:
        pass
    with open(best_params_path,'w') as fh:
        json.dump({'best_score': float(best_score),'best_params': best_params},fh,indent=2)
    return best_params


def train_eval_neural_cf(train_df,test_df,meta,epochs=5,batch_size=1024):
    if not HAS_TF:
        return None,{}
    num_users = meta['max_user_id']; num_items = meta['max_item_id']
    embed_dim = 64
    from tensorflow.keras import layers,models,callbacks,optimizers
    user_in = layers.Input(shape=(1,),name='user_in')
    item_in = layers.Input(shape=(1,),name='item_in')
    u_emb = layers.Embedding(input_dim=num_users+1,output_dim=embed_dim,name='u_emb')(user_in)
    i_emb = layers.Embedding(input_dim=num_items+1,output_dim=embed_dim,name='i_emb')(item_in)
    u_vec = layers.Flatten()(u_emb); i_vec = layers.Flatten()(i_emb)
    x = layers.Concatenate()([u_vec,i_vec])
    x = layers.Dense(256,activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128,activation='relu')(x)
    out = layers.Dense(5,activation='softmax')(x)
    model = models.Model([user_in,item_in],out)
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    tdf = train_df
    train_users = tdf['user_id'].values - 1
    train_items = tdf['item_id'].values - 1
    train_labels = tdf['rating'].values - 1
    es = callbacks.EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True)
    model.fit([train_users,train_items],train_labels,validation_split=0.05,epochs=epochs,batch_size=batch_size,callbacks=[es],verbose=0)
    ttdf = test_df
    preds_prob = model.predict([ttdf['user_id'].values - 1,ttdf['item_id'].values - 1],batch_size=4096)
    preds = np.argmax(preds_prob,axis=1) + 1
    acc = accuracy_score(ttdf['rating'].values,preds)
    f1m = f1_score(ttdf['rating'].values,preds,average='macro')
    logger.info(f"NeuralCF acc={acc:.4f} f1={f1m:.4f}")
    return model,{'accuracy': acc,'f1_macro': f1m}
def train_eval_siamese(train_df, test_df, meta, epochs=6, batch_size=1024):
    if not HAS_TF:
        return None, {}

    n_users = meta["max_user_id"]
    n_items = meta["max_item_id"]
    d_model = 64
    user_input = layers.Input(shape=(1,))
    item_input = layers.Input(shape=(1,))
    user_embedding = layers.Embedding(n_users + 1, d_model)(user_input)
    item_embedding = layers.Embedding(n_items + 1, d_model)(item_input)
    user_vec = layers.Flatten()(user_embedding)
    item_vec = layers.Flatten()(item_embedding)
    dense_a = layers.Dense(128, activation="relu")
    dense_b = layers.Dense(64, activation="relu")
    user_proj = dense_b(dense_a(user_vec))
    item_proj = dense_b(dense_a(item_vec))
    abs_diff = layers.Lambda(lambda z: tf.math.abs(z[0] - z[1]))([user_proj, item_proj])
    merged = layers.Concatenate()([user_proj, item_proj, abs_diff])
    hidden = layers.Dense(128, activation="relu")(merged)
    hidden = layers.Dropout(0.3)(hidden)
    output = layers.Dense(5, activation="softmax")(hidden)
    net = models.Model(inputs=[user_input, item_input], outputs=output)
    net.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    train_u = train_df["user_id"].to_numpy() - 1
    train_i = train_df["item_id"].to_numpy() - 1
    train_y = train_df["rating"].to_numpy() - 1
    stopper = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )
    net.fit(
        [train_u, train_i],
        train_y,
        validation_split=0.05,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[stopper],
        verbose=0,
    )
    test_u = test_df["user_id"].to_numpy() - 1
    test_i = test_df["item_id"].to_numpy() - 1
    prob = net.predict([test_u, test_i], batch_size=4096, verbose=0)
    pred = np.argmax(prob, axis=1) + 1
    acc = accuracy_score(test_df["rating"].to_numpy(), pred)
    f1m = f1_score(test_df["rating"].to_numpy(), pred, average="macro")
    logger.info(f"SiameseCF acc={acc:.4f} f1={f1m:.4f}")
    return net, {"accuracy": acc, "f1_macro": f1m}

#runner 
def run_all():

    full = pd.read_csv(os.path.join(dt_pth,"u.data"),sep='\t',names=['user_id','item_id','rating','timestamp'],engine='python')
    MAX_USER_ID = int(full['user_id'].max()); MAX_ITEM_ID = int(full['item_id'].max())
    logger.info(f"Detected user_id max = {MAX_USER_ID},item_id max = {MAX_ITEM_ID}")

    model_zoo = make_model_zoo()
    records = []
    per_model_scores = defaultdict(list)

    tuned_models_set = ['RandomForest','LightGBM','XGBoost'] 

    for fold in flds:
        logger.info(f"===== FOLD {fold} =====")
        train_path = os.path.join(dt_pth,f"u{fold}.base")
        test_path  = os.path.join(dt_pth,f"u{fold}.test")
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Missing fold files in dt_pth")
        train_df = pd.read_csv(train_path,sep='\t',names=['user_id','item_id','rating','timestamp'],engine='python')
        test_df  = pd.read_csv(test_path, sep='\t',names=['user_id','item_id','rating','timestamp'],engine='python')

        # Baseline
        mu,bu_map,bi_map = fit_baseline_biases(train_df,reg=10.0,n_iter=10)
        preds_base = []
        for _,r in test_df.iterrows():
            phat = mu + bu_map.get(int(r['user_id']),0.0) + bi_map.get(int(r['item_id']),0.0)
            phat = int(round(phat)); phat = max(1,min(5,phat))
            preds_base.append(phat)
        acc_base = accuracy_score(test_df['rating'].values,preds_base)
        f1_base = f1_score(test_df['rating'].values,preds_base,average='macro')
        logger.info(f"Baseline acc={acc_base:.4f},f1={f1_base:.4f}")
        records.append({'model':'Baseline','fold':fold,'accuracy':acc_base,'f1_macro':f1_base})

        # build features - cached
        X_train,X_test,y_train,y_test,X_train_nb,X_test_nb,meta = build_features_train_test_cached(train_df,test_df,ltdim,fold,cache_dir="fold_cache1")
        # print("Train shape:",X_train.shape)
        # print(X_train[:1])


        # exit(0)
        # iterate models
        for name,estimator in list(model_zoo.items()):
            t0 = time.time()
            try:
                best_params_path = os.path.join(sv_dir,f"best_params_{name}.json")
                to_tune = False
                if tuner and (fold == flds[0]) and (name in tuned_models_set) and (not os.path.exists(best_params_path)):
                    to_tune = True

                if os.path.exists(best_params_path) and not to_tune:
                    try:
                        with open(best_params_path,'r') as fh:
                            j = json.load(fh)
                            bp = j.get('best_params',{})
                    except Exception:
                        bp = {}
                    if bp:
                        est = clone(estimator)
                        try:
                            est.set_params(**bp)
                        except Exception:
                            logger.warning(f"Could not set some params from saved file for {name}")
                        logger.info(f"{name}: using saved best_params from {best_params_path}")
                        if name == 'XGBoost' and hasattr(est,'fit') and 'use_label_encoder' in getattr(est,'get_params',lambda: {})():
                            pass
                    else:
                        est = clone(estimator)
                elif to_tune:
                    try:
                        best_params = tune_model(name,estimator,X_train,y_train,n_trials=trls,cv=cross_v,n_jobs=naukri_jbs)
                        if best_params:
                            est = clone(estimator)
                            try:
                                est.set_params(**best_params)
                            except Exception:
                                logger.warning(f"Could not set some params for {name} after tuning")
                        else:
                            est = clone(estimator)
                    except Exception as e:
                        logger.exception(f"Tuning failed for {name}: {e}")
                        est = clone(estimator)
                else:
                    est = clone(estimator)

                logger.info(f"[{name}] starting fit (fold {fold})")
                fit_t0 = time.time()
                if name == 'ComplementNB':
                    est.fit(X_train_nb,y_train)
                    preds = est.predict(X_test_nb)
                elif name == 'XGBoost' and HAS_XGB:
                    est.fit(X_train,y_train - 1)
                    preds = est.predict(X_test) + 1
                else:
                    est.fit(X_train,y_train)
                    preds = est.predict(X_test)
                fit_dur = time.time() - fit_t0
                logger.info(f"[{name}] finished fit in {fit_dur:.1f}s")

                # scoring
                acc = accuracy_score(y_test,preds)
                f1m = f1_score(y_test,preds,average='macro')
                prec = precision_score(y_test,preds,average='macro',zero_division=0)
                rec = recall_score(y_test,preds,average='macro',zero_division=0)
                logger.info(f"{name:15s} acc={acc:.4f} f1={f1m:.4f} total_time={time.time()-t0:.1f}s")
                records.append({'model':name,'fold':fold,'accuracy':acc,'f1_macro':f1m,'precision_macro':prec,'recall_macro':rec})
                per_model_scores[name].append(acc)
                # save artifact for this fold
                try:
                    joblib.dump(est,os.path.join(sv_dir,f"{name}_fold{fold}.joblib"))
                except Exception:
                    logger.warning(f"Could not save {name} artifact for fold {fold}")
                # save classification report
                cr = classification_report(y_test,preds,output_dict=True,zero_division=0)
                with open(os.path.join(rpt,f"{name}_fold{fold}.json"),'w') as fh:
                    json.dump({'metrics': {'accuracy': acc,'f1_macro': f1m,'precision_macro': prec,'recall_macro': rec},'classification_report': cr},fh,indent=2)
            except Exception as e:
                logger.exception(f"Error training/evaluating {name} on fold {fold}: {e}")
                records.append({'model': name,'fold': fold,'accuracy': math.nan,'f1_macro': math.nan,'precision_macro': math.nan,'recall_macro': math.nan})
                per_model_scores[name].append(math.nan)
        if HAS_TF:
            # print(meta)
            try:
                if cf_Chk:
                    model_cf,metrics_cf = train_eval_neural_cf(train_df,test_df,meta,epochs=5)
                    if metrics_cf:
                        records.append({'model':'NeuralCF','fold':fold,'accuracy':metrics_cf['accuracy'],'f1_macro':metrics_cf['f1_macro']})
                if siamese_chk:
                    model_siam,metrics_siam = train_eval_siamese(train_df,test_df,meta,epochs=6)
                    if metrics_siam:
                        records.append({'model':'SiameseCF','fold':fold,'accuracy':metrics_siam['accuracy'],'f1_macro':metrics_siam['f1_macro']})
            except Exception as e:
                logger.exception("TF model training error: " + str(e))

    # save results & summary
    df_results = pd.DataFrame(records)
    df_results.to_csv(rscsv,index=False)
    logger.info(f"Saved per-fold metrics to {rscsv}")

    logger.info("=== AVERAGE SCORES ACROSS flds ===")
    summary = []
    for m in sorted(df_results['model'].unique()):
        sub = df_results[df_results['model'] == m]
        avg_acc = sub['accuracy'].mean()
        avg_f1 = sub['f1_macro'].mean()
        avg_prec = sub['precision_macro'].mean() if 'precision_macro' in sub.columns else None
        avg_rec = sub['recall_macro'].mean() if 'recall_macro' in sub.columns else None
        logger.info(f"{m:15s}  acc={avg_acc:.4f}  f1_macro={avg_f1:.4f}  prec_macro={avg_prec}  rec_macro={avg_rec}")
        summary.append({'model':m,'acc_mean':float(avg_acc),'f1_macro_mean':float(avg_f1),'precision_macro_mean':float(avg_prec) if avg_prec is not None else None,'recall_macro_mean':float(avg_rec) if avg_rec is not None else None})
    pd.DataFrame(summary).to_csv(os.path.join(sv_dir,"ml100k_avg_metrics_by_model_no_meta.csv"),index=False)
    logger.info("Saved aggregated model averages")
    logger.info("Done.")

if __name__ == "__main__":
    run_all()
