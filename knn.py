#-------------------------------------------------------------------------
# AUTHOR: Grecia Alvarado
# FILENAME: knn.py
# SPECIFICATION: Leave-one-out and determine error rate for 1NN
# FOR: CS 4210- Assignment #2
# TIME SPENT: 40 minutes
#-----------------------------------------------------------*/
#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH 
#AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays
#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv
db = []
numWrong = 0
#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader): #i is 0-10, so line no in csv
      if i > 0: #skipping the header
         db.append (row)
for i, instance in enumerate(db): #instance is the array of the line
    #transform the original training classes to numbers and add them to the vector 
    #Y. Do not forget to remove the instance that will be used for testing in this 
    #iteration.
    #For instance, Y = [1, 2, ,...]. Convert values to float to avoid warning 
    #messages
    X = []
    Y = []
    temp = []
    test = []
    for j in range(0,len(instance)):
        if j == len(instance)-1:
            if instance[j] == "-":
                Y.append(0.0)
            elif instance[j] == "+":
                Y.append(1.0)
        else:
            temp.append(float(instance[j]))
    
    if i != len(db)-1:
        test.append(float(db[i+1][0]))
        test.append(float(db[i+1][1]))
        if db[i+1][2] == "-":
            test.append(0.0)
        elif db[i+1][2] == "+":
            test.append(1.0)
    else:
        test.append(float(db[0][0]))
        test.append(float(db[0][1]))
        if db[0][2] == "-":
            test.append(0.0)
        elif db[0][2] == "+":
            test.append(1.0)
    X.append(temp)

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)
    #use your test sample in this iteration to make the class prediction. For 
#instance:
    class_predicted = clf.predict([[test[0], test[1]]])[0]

    #compare the prediction with the true label of the test instance to start 
#calculating the error rate.
    if class_predicted != int(test[2]):
        numWrong += 1
#print the error rate
errorRate = numWrong/len(db)
print("Error rate: ", errorRate)