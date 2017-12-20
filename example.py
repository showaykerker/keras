
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from data_gen.generator import Generator


if __name__ == '__main__':

	n_data = 50000
	n_train = n_data*0.8

	g = Generator(n_data=n_data)
	#X, Y = g.create_data()
	#g.save_data(name='Data')
	X, Y = g.read_data(name='Data')
	print(X)
	print(Y)
	
	X_train, Y_train = X[:int(n_train)], Y[:int(n_train)]
	X_test, Y_test = X[int(n_train):], Y[int(n_train):]

	# Build Model
	model = Sequential()
	model.add(Dense(units=1, input_dim=1))
	model.compile(loss='mse', optimizer='sgd')

	# Another Way to Build Model
	model2 = Sequential([
		Dense(units=1, input_dim=1)
	])
	model2.compile(loss='mse', optimizer='sgd')	


	# Train Models
	print('\nModel1: Trainning On Batch ------------')
	for i in range(501):
		cost = model.train_on_batch(X_train, Y_train)
		if i % 200 == 0:
			print('step %i, train cost ' % i , cost)

	print('\nModel2: Fitting Model ------------')
	model2.fit(X_train, Y_train, epochs=32, steps_per_epoch=64, shuffle=True)
		


	# Test Models
	print('\n\nModel1 Testing --------------')
	cost = model.evaluate(X_test, Y_test, batch_size=int(n_data*0.8/10))
	print('\nModel2 Testing --------------')
	cost2 = model2.test_on_batch(X_test, Y_test)

	# Show Results
	print('\ncost =', cost, ',', cost2)
	W1, b1 = model.layers[0].get_weights()
	W2, b2 = model2.layers[0].get_weights()
	print(W1, b1)
	print(W2, b2)
	
	Y_pred = model.predict(X_test)
	Y_pred2 = model2.predict(X_test)
	plt.scatter(X_test, Y_test, s=0.1)
	plt.plot(X_test, Y_pred, 'go', ms=0.3)
	plt.plot(X_test, Y_pred2, 'ro', ms=0.3)
	plt.show()