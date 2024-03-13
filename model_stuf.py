from sklearn.metrics import roc_auc_score
import xgboost as xgb
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from data import *
import optuna
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import math


def w2v(df:pd.DataFrame,vector_size):
    #win_label = df['radiant_win'].apply(lambda x: 999 if x > 0 else -999)
    df = df.drop(['radiant_win','id'], axis='columns')

    X1 = df.iloc[:, :5].copy()
    #X1['radiant_win'] = win_label
    X2 = df.iloc[:, 5:].copy()
    #X2['radiant_win'] = win_label * -1
    sentences = []
    for _, sentence in X1.iterrows():
        sentence = list(sentence)
        sentences.append(sentence)
    for _, sentence in X2.iterrows():
        sentence = list(sentence)
        sentences.append(sentence)
    model = Word2Vec(sentences, min_count=1, vector_size=vector_size, window=5, sg=1, epochs=40)
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
    X, Y = get_data()
    print(len(X['old_pro']),len(Y['old_pro']))
    try:
        current_model = xgb.Booster()  # init model
        current_model.load_model('model.json')
        current_model_scores = gini_scores(current_model, X, Y, 'Current model')
    except:
        pass

    def objective(trial):
        combined_df = pd.concat([X['old_pro'], Y['old_pro']], axis=1)
        # Identify duplicates and non-duplicates
        duplicates_df = combined_df[combined_df.duplicated(keep='first')]
        print(duplicates_df.head())
        non_duplicates_df = combined_df.drop_duplicates(keep=False)
        # Split non-duplicates into training and test sets
        train_unique, test_unique = train_test_split(non_duplicates_df, test_size=0.2)
        # Append the duplicates to the training set
        train_df = pd.concat([train_unique, duplicates_df])
        # Ensure test set only contains unique instances
        test_df = test_unique
        # Separate features and target for both training and test sets
        x = train_df.drop(columns=['radiant_win'])  # Replace 'target_column' with your actual target column name
        y = train_df['radiant_win']  # Replace 'target_column' with your actual target column name

        xv = test_df.drop(columns=['radiant_win'])  # Replace 'target_column' with your actual target column name
        yv = test_df['radiant_win']  # Replace 'target_column' with your actual target column name
        print(len(x),len(xv))

        dtest = xgb.DMatrix(xv, label=yv)
        dtrain = xgb.DMatrix(x, label=y)
        param = {
            "verbosity": 0,
            "objective": trial.suggest_categorical('objective',["binary:logistic"]),
            "eval_metric": trial.suggest_categorical("eval_metric",["auc"]),
            "booster": trial.suggest_categorical('booster',["gbtree"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 1, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1, log=True),
        }

        if param["booster"] == "gbtree":
            param["max_depth"] = trial.suggest_int("max_depth", 1,16)
            param["eta"] = trial.suggest_float("eta", 1e-5, .3, log=True)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        # Add a callback for pruning.
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
        bst = xgb.train(param, dtrain, evals=[(dtest,'validation')], callbacks=[pruning_callback])
        test_preds = bst.predict(dtest)
        gini_coefficient = 2 * roc_auc_score(yv, test_preds) - 1
        return gini_coefficient

    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction='maximize')
    study.optimize(objective, n_trials=100)
#
    print(study.best_trial.params)
    param = study.best_trial.params

    new_gini, model = gini_on_folds(X['old_pro'], Y['old_pro'], param)

    new_model_scores = gini_scores(model, X, Y, 'new model')

    new_gini,model = gini_on_folds(X['Big'],Y['Big'],param)

    new_model_scores = gini_scores(model,X,Y,'new model')

    if new_model_scores[3]>current_model_scores[3]:
        train3 = xgb.DMatrix(X['Big'], label=Y['Big'])
        new_model = xgb.train(param, train3)
        new_model.save_model('model.json')
    #model.save_model('model.json')

    #try:
    #    if new_model_scores[3]>current_model_scores[3]:
    #        print('New better')
    #        train3 = xgb.DMatrix(X['Big'], label=Y['Big'])
    #        new_model = xgb.train(param,train3)
    #        new_model.save_model('model.json')
    #    else:
    #        print('old better')
    #        train3 = xgb.DMatrix(X['Big'], label=Y['Big'])
    #        new_model = xgb.train(param,train3)
    #        new_model.save_model('model.json')
#
    #except:
    #    print('old does not exist')
    #    train3 = xgb.DMatrix(X['Big'], label=Y['Big'])
    #    new_model = xgb.train(param,train3)
    #    new_model.save_model('model.json')
    return X,Y


def g(RD):
    """
    Calculate the g(RD) function used in the Glicko-2 rating system.

    :param RD: Rating deviation of the player
    :return: The result of the g(RD) function
    """
    q = math.log(10) / 400  # System constant
    return 1 / math.sqrt(1 + 3 * q ** 2 * RD ** 2 / (math.pi ** 2))


def glicko2_win_prob(rating_A, rating_B, RD_A, RD_B):
    """
    Calculate the win probability of Player A vs. Player B given their Glicko-2 ratings and rating deviations.

    :param rating_A: Glicko-2 rating of Player A
    :param rating_B: Glicko-2 rating of Player B
    :param RD_A: Rating deviation of Player A
    :param RD_B: Rating deviation of Player B
    :return: Win probability of Player A
    """
    RD = (RD_A + RD_B) / 2  # Average rating deviation for the match
    g_RD = g(RD)
    exponent = (rating_B - rating_A) / (400 * g_RD)
    win_prob_A = 1 / (1 + 10 ** exponent)
    return win_prob_A

def gini_on_folds(X, Y, params, n_splits=10, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mean_gini = 0

    for num, (train_id, valid_id) in enumerate(kf.split(X)):
        X_train, X_valid = X.iloc[train_id], X.iloc[valid_id]
        y_train, y_valid = Y.iloc[train_id], Y.iloc[valid_id]
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_valid, label=y_valid)

        model = xgb.train(params, dtrain=train)

        test_preds = model.predict(valid)
        gini_coefficient = 2 * roc_auc_score(y_valid, test_preds) - 1
        print(f"Fold {num} | GINI: {gini_coefficient}")

        mean_gini += gini_coefficient / n_splits

    print(f"\nOverall Gini: {mean_gini}")
    return mean_gini, model


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
    plot_region(x_bounds=(-928.0, 928.0), y_bounds=(-928.25, 928))#all now
    plt.show()


def get_glicko_prob(A_name,B_name):
    ratings = pd.read_csv('pro_ratings.csv')
    ra, raD = ratings[ratings.teamName == A_name][['glicko2.rating', 'glicko2.phi']].values[0] if A_name in ratings.teamName.values else (1500, 350)
    rb, rbD = ratings[ratings.teamName == B_name][['glicko2.rating', 'glicko2.phi']].values[0] if B_name in ratings.teamName.values else (1500, 350)
    return glicko2_win_prob(ra, rb, raD, rbD)