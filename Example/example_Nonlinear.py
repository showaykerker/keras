
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
from data_gen.generator import Generator


if __name__ == '__main__':

	n_DATA = 5000
	n_TRAIN = int(n_DATA*0.8)
	LINEAR = False
	GENERATE_DATA = False

	X, Y = None, None


	g = Generator(n_data=n_DATA)
	if GENERATE_DATA:
		X, Y = g.create_data()
		g.save_data(name='Data')
	else: 
		X, Y = g.read_data(name='Data')
	
	X_train, Y_train = X[:n_TRAIN], Y[:n_TRAIN]
	X_test, Y_test = X[n_TRAIN:], Y[n_TRAIN:]

	
	
	if LINEAR:
		# For Linear Model
		# Build Model
		model = Sequential()
		model.add(Dense(units=1, input_dim=1))
		model.compile(loss='mse', optimizer='sgd')

		# Another Way to Build Model
		model2 = Sequential([
			Dense(units=1, input_dim=1)
		])
		model2.compile(loss='mse', optimizer='sgd')	
	else:
		# For Non-Linear Model
		model = Sequential([
			Dense(100, input_dim=1, activation='tanh'),
			Dense(100, activation='relu'),
			Dense(1)
		])
		model.compile(loss='mse', optimizer='sgd')

		model2 = Sequential()
		model2.add(Dense(100, input_dim=1))
		model2.add(Activation('tanh'))
		model2.add(Dense(100))
		model2.add(Activation('relu'))
		model2.compile(loss='mse', optimizer='sgd')
		# Train Models


	print('\nModel1: Trainning On Batch ------------')
	for i in range(7001):
		cost = model.train_on_batch(X_train, Y_train)
		if i % 2000 == 0:
			print('step %i, train cost ' % i , cost)

	print('\nModel2: Fitting Model ------------')
	model2.fit(X_train, Y_train, epochs=128, steps_per_epoch=64, shuffle=True)
		

	# Test Models
	print('\n\nModel1 Testing --------------')
	cost = model.evaluate(X_test, Y_test, batch_size=int(n_DATA*0.8/10))
	print('\nModel2 Testing --------------')
	cost2 = model2.test_on_batch(X_test, Y_test)

	# Show Results
	print('\ncost =', cost, ',', cost2)
	if LINEAR:
		W1, b1 = model.layers[0].get_weights()
		W2, b2 = model2.layers[0].get_weights()
		print(W1, b1)
		print(W2, b2)
	
	Y_pred = model.predict(X_test)
	Y_pred2 = model2.predict(X_test)
	plt.scatter(X_test, Y_test, s=0.1)
	plt.plot(X_test, Y_pred, 'go', ms=0.3)
	plt.plot(X_test, Y_pred2, 'ro', ms=0.3)
	if GENERATE_DATA: plt.plot(g.X, g.Target, 'bo', ms=0.1)
	plt.show()