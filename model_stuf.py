import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from data import *
import optuna
import joblib
from gensim.models import KeyedVectors
from sklearn.utils.multiclass import type_of_target


def w2v(df,vector_size):
    df[:11] = df[:11].astype(int)
    #X = df.drop(['radiant_win'], axis='columns')
    X1 = df.iloc[:, :5].copy()
    X1['radiant_win'] = df['radiant_win'].astype(str)
    X2 = df.iloc[:, 5:].copy()
    X2['radiant_win'] = df['radiant_win'].astype(str)
    sentences = []
    for _, sentence in X1.iterrows():
        sentence = list(sentence)
        sentences.append(sentence)
    for _, sentence in X2.iterrows():
        sentence = list(sentence)
        sentences.append(sentence)
    model = Word2Vec(sentences, min_count=1, vector_size=vector_size, window=5, sg=0, epochs=40)
    word_vectors = model.wv
    return word_vectors


def get_gini(model, x, y, verbose=False, model_name = None, data_name = None):
    dvalid = xgb.DMatrix(x, label=y)
    test_preds = model.predict(dvalid)
    gini_coefficient = 2 * roc_auc_score(y, test_preds) - 1
    if verbose:
        print(f'{model_name} on {data_name} gini: {round(gini_coefficient,3)}')
    return gini_coefficient

def gini_scores(model,X,Y,model_name):
    ginis=[]
    for z in X:
        y=Y[z]
        x=X[z]
        data_name=z
        ginis.append(get_gini(model,x,y,verbose=True,model_name=model_name,data_name=data_name))
    return ginis

def update_model():
    X, Y = read_data()
    try:
        current_model = xgb.Booster()  # init model
        current_model.load_model('model.json')
        current_model_scores = gini_scores(current_model, X, Y, 'Current model')
    except:
        pass

    def objective(trial):
        dtest = xgb.DMatrix(X['old_pro'], label=Y['old_pro'])
        dtrain = xgb.DMatrix(X['old_ladder'], label=Y['old_ladder'])
        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        }

        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        # Add a callback for pruning.
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
        bst = xgb.train(param, dtrain, evals=[(dtest,'validation')], callbacks=[pruning_callback])
        test_preds = bst.predict(dtest)
        gini_coefficient = 2 * roc_auc_score(Y['old_pro'], test_preds) - 1
        return gini_coefficient

    #study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction='maximize')
    #study.optimize(objective, n_trials=100)
#
    #print(study.best_trial)
    #parames = study.best_trial.params
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": "gbtree",
    }
    train = xgb.DMatrix(X['old_ladder'], label=Y['old_ladder'])
    #train2 = xgb.DMatrix(X['old_pro'], label=Y['old_pro'])
    new_model = xgb.train(param,dtrain = train)

    new_model_scores = gini_scores(new_model,X,Y,'New model')

    try:
        if new_model_scores[3]>current_model_scores[3]:
            print('New better')
            train3 = xgb.DMatrix(X['Big'], label=Y['Big'])
            new_model = xgb.train(param,train3)
            new_model.save_model('model.json')
        else:
            print('old better')
            train3 = xgb.DMatrix(X['Big'], label=Y['Big'])
            new_model = xgb.train(param,train3)
            new_model.save_model('model.json')

    except:
        print('old does not exist')
        train3 = xgb.DMatrix(X['Big'], label=Y['Big'])
        new_model = xgb.train(param,train3)
        new_model.save_model('model.json')


def qwe(wv):
    tsne = TSNE(n_components=2, random_state=0)
    word_vectors_matrix_2d = tsne.fit_transform(wv.vectors)
    points = pd.DataFrame([(word, coords[0], coords[1]) for word, coords in
                           [(word, word_vectors_matrix_2d[wv.key_to_index[word]]) for word in wv.key_to_index]],
                          columns=["word", "x", "y"]
                          )
    Localized_hero_list = get_Localized_hero_list()
    def plot_region(x_bounds, y_bounds):
        slice = points[
            (x_bounds[0] <= points.x) &
            (points.x <= x_bounds[1]) &
            (y_bounds[0] <= points.y) &
            (points.y <= y_bounds[1])
            ]
        ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
        for i, point in slice.iterrows():
            try:
                ax.text(point.x + 0.005, point.y + 0.005, list(filter(lambda Localized_hero_list: Localized_hero_list['id'] == int(point.word), Localized_hero_list))[0]['name'][14:], fontsize=11)
            except:
                pass
    plot_region(x_bounds=(-28.0, 28.0), y_bounds=(-28.25, 28))#all now
    plt.show()