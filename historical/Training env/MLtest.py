import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1)) #mora bit z rowi
y = np.array([5, 20, 14, 32, 22, 38])

#https://scikit-learn.org/stable/modules/ensemble.html#forest

'''
za mojo stvar rabim dat nevem 
x = [[close diff, open diff]]
'''

model = LinearRegression()
model.fit(x,y)

r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}") #to je R**2 is the proportion of variation in the dependant variable

y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")

'''
to je linear regression, obstaja tudi gradient voost (xgboost) dober kao za tabular data
xdboost je train small shallow trees, they learn what the prev one went wrong
basically policy gradient

ltsm, gru, cnn so neural nets ampak overfitting

random forrest (many decision treesv)
yes or no questions with greedy decision
veliko dreves
avg of all trees je final prediciton
'''

