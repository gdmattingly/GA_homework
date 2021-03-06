import numpy as np
import csv as csv

#Import Data, read it into an object, grab the header, and read the rows into 'data'
csv_file_object = csv.reader(open('/home/GA11/data/titanic/train.csv','rb'))
header = csv_file_object.next()
data = []
for row in csv_file_object:
   data.append(row)
data = np.array(data)

#Calclulate some summary stats
number_passengers = np.size(data[0::,0].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

print '---- Summary Stats ----'
print 'Number of passengers in training set = %s' % number_passengers
print 'Number of Survivors in training set = %sw' % number_survived
print 'Survival Rate in training set = %s' % proportion_survivors

#Calculate some gender based stats
women_only_stats = data[0::,4] =='female'
men_only_stats = data[0::,4] == 'male'
unknown_gender_stats = data[::,4] == ""

women_onboard_idx = data[women_only_stats,0].astype(np.float)
men_onboard_idx = data[men_only_stats,0].astype(np.float)

print women_onboard_idx

#number_women_onboard = np.stats(data[women_onboard_idx

passengers_missing_age = data[0::,5] == ""
count_missing_age = np.size(data[passengers_missing_age,0].astype(np.float))
sum_passenger_age = np.sum(data[0::,5].astype(np.float))
avg_age = sum_passenger_age / (number_passengers - count_missing_age)
print avg_age
