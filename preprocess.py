import csv
import numpy as np
from tqdm import tqdm
import sys
data = []
img_list = []
with open('fer2013mask.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for i, row in tqdm(enumerate(csv_reader)):
        if i == 0:
            continue
        str_list = list(row[1].split(" "))
        index = int(row[0])
        img = np.array([float(v) for v in str_list])
#        img_list.append([index,img])
        usage = row[2]
        data.append([index, img, usage])
#        if i > 100:
#            break
emotion = []
#cmpi_list = []
with open('fer2013.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=",")
    for i,row in tqdm(enumerate(csv_reader)):
        if i == 0:
            emotion.append('none')
#            cmpi_list.append('dog')
            continue
#        cmpi_list.append(np.array([float(v) for v in list(row[1].split(" "))]))
        
        emotion.append(row[0])
#        if i > 200:
#            break
#print(len(img_list), len(cmpi_list))
#for img in img_list:
#    print(img[0],(cmpi_list[img[0]] - img[1]).sum())
#sys.exit(0)
with open('fer2013maskRight.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['emotion', 'pixels', 'Usage'])
    for row in tqdm(data):
        writer.writerow([emotion[row[0]], " ".join(str(int(i)) for i in row[1]), row[2]])
