from sklearn.linear_model import LogisticRegression
import pandas as pd
# using same iris data split used for OneVsRestClassifier part1 code
# creating dummy variable for all target class
dummy_y = label_binarize(y_train, classes = [0,1,2])
# Fitting model 
model_0 = LogisticRegression().fit(X_train,dummy_y[:,0])
model_1 = LogisticRegression().fit(X_train,dummy_y[:,1])
model_2 = LogisticRegression().fit(X_train,dummy_y[:,2])
# getting probility of True only i.e class 1 
model_0_max_p = model_0.predict_proba(X_test)[:,1:3]
model_1_max_p = model_1.predict_proba(X_test)[:,1:3]
model_2_max_p = model_2.predict_proba(X_test)[:,1:3]
# creating dataframe
df_pred_prob = pd.DataFrame(np.hstack([model_0_max_p, model_1_max_p, model_2_max_p]), columns = [0,1,2])
# generating label
df_pred_prob['final_label'] = df_pred_prob.idxmax(axis=1)
