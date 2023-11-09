from sklearn.ensemble import GradientBoostingRegressor
import shap
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost


df = pd.DataFrame({'target':[23,42,58,29,28],
                      'feature_1' : [38, 83, 38, 28, 57],
                      'feature_2' : ['A', 'B', 'A', 'C','A']
                  })

df["feature_1"]=df["feature_1"].astype(int)
df["target"]=df["target"].astype(int)
df["target"]=df["target"].astype('category')

print(df)
SEED=42
model = xgboost.XGBRegressor(enable_categorical=True,
                                       tree_method='hist')

scale= preprocessing.StandardScaler()

#X=df[["feature_1","feature_2"]]
columns=["feature_1","feature_2"]
n_features=len(columns)
X=np.array(df[columns]).reshape(-1,n_features)
y=np.array(df["target"])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
model.fit(X_train,y_train)






explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test ) 



print(shap_values)


y_pred=model.predict(X_test)

x=np.arange(len(X_test))
plt.bar(x,y_test)
plt.bar(x,y_pred,color='green')
plt.show()


'my_col'[0:1]







df = pd.DataFrame({'target':[0,1,3,4,2],
                      'feature_1' : [4, 5, 6, 4, 6],
                      'feature_2' : ['AJDJDNDLS', 'BSKJDJJDDJD', 'ADJJDJDJD', 'CDIDJDJJD','ADIJDJDJDJ']
                  })

df['new'] = df.apply(
    lambda x: x['feature_2'][x['target']:x['feature_1']], axis=1)



row["source"][item["startOffset"] : item["endOffset"]]

