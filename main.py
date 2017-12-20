
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from data_gen.generator import Generator


if __name__ == '__main__':

	n_DATA = 5000
	n_TRAIN = int(n_DATA*0.8)

	X, Y = None, None

	g = Generator()

	X, Y = g.read_data(name='Data-linear')
	
	X_train, Y_train = X[:n_TRAIN], Y[:n_TRAIN]
	X_test, Y_test = X[n_TRAIN:], Y[n_TRAIN:]


	# Build Model
	

	
	

	# Train Model
	print('\nTrainning ------------')
	

	
		

	# Test Models
	print('\n\nTesting --------------')
	cost = model.evaluate(X_test, Y_test, batch_size=int(n_DATA*0.8/10))
	
	# cost = model.test_on_batch(X_test, Y_test) 


	# Show Results
	print('\ncost =', cost, ',', cost2)
	W1, b1 = model.layers[0].get_weights()
	W2, b2 = model2.layers[0].get_weights()
	print(W1, b1)
	print(W2, b2)
	
	# Plot Results
	Y_pred = model.predict(X_test)
	Y_pred2 = model2.predict(X_test)
	plt.scatter(X_test, Y_test, s=0.1)
	plt.plot(X_test, Y_pred, 'go', ms=0.4)
	plt.plot(X_test, Y_pred2, 'ro', ms=0.3)
	plt.show()