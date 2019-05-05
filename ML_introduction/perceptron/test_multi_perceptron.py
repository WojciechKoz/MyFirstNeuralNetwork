from perceptron import Perceptron
from initialization_objects import create_data, prepare_data
from multi_perceptron import Classifier


groups = create_data([(0,0), (10, 0), (0, 5)], 1)

X, Y = prepare_data(groups)

model = Classifier(["A", "B", "C"])

model.fit_perceptrons(X, Y)

model.show_results()
