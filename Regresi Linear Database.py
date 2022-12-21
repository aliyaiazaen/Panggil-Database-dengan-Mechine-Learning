import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

#Database
# x = Data, y = Target
#x = [[1], [2], [5], [8], [9], [11], [13], [16], [18], [20]]
#y = [3, 6, 15, 24, 27, 33, 39, 48, 54, 60]
FileDB = 'perkalian.txt'
Database = pd.read_csv(FileDB, sep=" ", header=0)
print ("---------------------") 
print (Database)
#x = data, y = target
x = Database [[u'x']] #ciri1, ciri2, dst 
y = Database.Target

regr = LinearRegression().fit(x,y)
regr.score(x, y)

#Data uji
predict = np.array([[2002 ]])

#Menampilkan data prediksi
print ("Prediksi")
print ("Input = ", predict)
print ("Output = ", regr.predict(predict))
