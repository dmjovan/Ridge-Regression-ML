from custom_functions.functions import *


def model(X: np.ndarray) -> np.ndarray:
    
	"""
		Funkcija za vrsenje predikcija pomocu 'istreniranog' 
		modela.

		:params:
			- X: matrica dimenzija 5xN, sa N primera i 5 prediktora

		:return:
			- y_hat: predikcija izlaza
 	"""

	theta, theta0 = load_params()
	y_hat = estimate_new_data(X, theta, theta0)
    
	return y_hat
