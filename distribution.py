import csv
import numpy as np
from tqdm import tqdm
expression_train = np.zeros(7)
expression_val = np.zeros(7)
expression_test = np.zeros(7)
with open('fer2013maskFormal.csv','r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=",")
    for i, row in tqdm(enumerate(csv_reader)):
        if i == 0:
            continue
        if row[2] == 'Training':
            expression_train[int(row[0])] += 1
        elif row[2] == 'PublicTest':
            expression_test[int(row[0])] += 1
        else:
            expression_val[int(row[0])] += 1
print(expression_train / expression_train.sum())
print(expression_val / expression_val.sum())
print(expression_test / expression_test.sum())
