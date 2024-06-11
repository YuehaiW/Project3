import numpy as np
import pandas as pd
import xgboost as xgb
# 可视化
import seaborn as sns
import matplotlib.pyplot as plt
# 预处理，模型
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler


# 计算常用指标
def compute_indexes(tp, fp, tn, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = (2 * precision * recall) / (precision + recall)
    return accuracy, precision, recall, F1


df = pd.read_parquet('ST data.parquet')
data = df.dropna(axis=0)
train, test = data[data.date <= '20191231'], data[data.date >= '20200101']
X_train, y_train = train.drop(['date', 'code', 'is_st'], axis=1), train['is_st']
X_test, y_test = test.drop(['date', 'code', 'is_st'], axis=1), test['is_st']

sampler = RandomUnderSampler(random_state=123)
X_train_rsl, y_train_rsl = sampler.fit_resample(X_train, y_train)
parameters = {'n_estimators': 100, 'max_depth': 7, 'reg_alpha': 8, 'reg_lambda': 8, 'Gamma': 8}
model_xgb = XGBClassifier(n_estimators=100, max_depth=7, reg_alpha=8, reg_lambda=8, gamma=8,
                          objective="binary:logistic", eval_metric="auc")
model_xgb.fit(X_train_rsl, y_train_rsl)
y_pred_xgb = model_xgb.predict(X_test)
print("Test accuracy: {}".format(accuracy_score(y_test, y_pred_xgb)))
xgb_prob = model_xgb.predict_proba(X_test)
fpr_xgb, tpr_xgb, thr_xgb = roc_curve(y_test, xgb_prob[:, 1])
auc_xgb = auc(fpr_xgb, tpr_xgb)
print("AUC Score: {}".format(auc_xgb))
mat = confusion_matrix(y_test, y_pred_xgb)
tn, fp, fn, tp = mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1]
accuracy, precision, recall, F1 = compute_indexes(tp, fp, tn, fn)
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1 Score: {}".format(F1))

# 绘制特征重要性排序
xgb.plot_importance(model_xgb, max_num_features=8, importance_type='gain')
plt.show()
