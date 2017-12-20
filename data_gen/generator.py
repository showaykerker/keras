import numpy as np
import csv
import matplotlib.pyplot as plt # sudo -H pip3 install matplotlib

np.random.seed(1024)


class Generator:
	def __init__(self, n_data=5000):
		self.n_data=n_data

	def create_data(self, name='NewData',n_data = None, mean = 0, std = 0.15):
		if n_data is None: n_data = self.n_data
		self.X = np.linspace(-1, 1, n_data)
		np.random.shuffle(self.X)
		self.Y = (1.2 * self.X ** 3 - 0.0125 * self.X ** 2 + 0.04 * self.X - 0.05 + np.random.normal(mean, std, (n_data, )))/1.5
		#self.Y = 0.2 * self.X - 0.6 + np.random.normal(mean, std/5, (n_data,))
		self.Target = (1.2 * self.X ** 3 - 0.0125 * self.X ** 2 + 0.04 * self.X - 0.05 )/1.5

		return self.X, self.Y


	def save_data(self, name='NewData'):
		
		if not hasattr(self, 'X') or not hasattr(self, 'Y'): 
			input('No data')
			return False

		with open(name+'.csv', 'w') as f:
			wtr = csv.writer(f)
			for i in range(len(self.X)):
				wtr.writerow((self.X[i], self.Y[i]))
	
	def read_data(self, name='NewData'):

		with open(name+'.csv') as f:
			rdr = csv.reader(f)
			rX = []
			rY = []
			for x, y in rdr:
				rX.append(float(x))
				rY.append(float(y))

			rX = np.array(rX, np.float32)
			rY = np.array(rY, np.float32)

		return rX, rY

	def plot_data(self, X = None, Y = None):
		if X is None and Y is None:
			plt.scatter(self.X, self.Y, s=0.1)
			plt.show()
		else:
			plt.scatter(X, Y, s=0.1)
			plt.show()


if __name__ == '__main__':
	g = Generator()
	g.create_data()
	g.read_data()
	g.plot_data()
