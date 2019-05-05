from init_obj import create_universe, prepare_data, show_universe, test_model
from adaline import Adaline
from adalineSGD import AdalineSGD
import numpy as np

groups = create_universe()
X, Y = prepare_data(groups)

# show_universe(groups)


y = np.where(Y == 'A', 1, -1)

model = Adaline(0.0001, 50)
model.fit(X, y)

# stochastic

model2 = AdalineSGD(0.0001, 50)

model2.fit(X, y)

test_model(X, model, model2)
