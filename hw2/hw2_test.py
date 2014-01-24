import numpy as np
from hw2_defs import load_iris_data, cross_validate, log_reg, get_c_values

# Load the Iris Data Set into XX for data, yy for targets, and y for labels
(XX, yy, y) = load_iris_data()

# arbitrarily setting number of folds at 3. Consider changing to user defined.
num_folds = 3

# setting tracking variables to zero
best_model_score=0
best_c=0

#print 'Please input upper and lower range limits for C:'
#c_upper = float(raw_input('Please enter the largest C value:'))
#c_lower = float(raw_input('Please input the smallest C value:'))
#c_interval = float(raw_input('Please input the increments of C to test within #this range:'))

#c_values = np.arange(c_lower,c_upper,c_interval,dtype=float)
#c_values = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]
c_values = get_c_values()

print c_values
#c_values = np.array(c_values.tolist())


for c_step in np.nditer(c_values):
    score_c = cross_validate(XX, yy, log_reg, num_folds, c_step)
    print "C value %s yields an accuracy of %s" % (c_step, score_c)
    if score_c > best_model_score:
        best_model_score = score_c
        best_c = c_step
    print ""
print 'Best C value is %s' % best_c
print 'It yields a model accuracy of %s' % best_model_score
