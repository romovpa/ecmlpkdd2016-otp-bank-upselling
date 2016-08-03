import pandas
import xgboost as xgb

features_train = pandas.read_csv('features/upselling_train.csv')
features_test = pandas.read_csv('features/upselling_test.csv')

sel_features = features_train.columns[4:]
X = features_train.ix[:, sel_features].values
X_test = features_test.ix[:, sel_features].values
y = features_train.ix[:, 'apply_future'].values

model = xgb.XGBClassifier().fit(X, y)
y_predicted = model.predict_proba(X_test)

pandas.Series(y_predicted[:, 1], name='SCORE', index=pandas.Index(features_test.user_id, name='#USER_ID'))\
    .to_csv('result/submission.csv', header=True)
    
    
# dump model for xgbfi

def create_feature_map(fmap_filename, features):
    outfile = open(fmap_filename, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
    
create_feature_map('result/model.fmap', sel_features) 
model.booster().dump_model('result/model.dump', fmap='model.fmap', with_stats=True)
