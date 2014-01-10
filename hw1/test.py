from hw1 import load_iris_data, cross_validate, knn, nb
# Imports functions and opject definitions from hw1.py

(iris_petal_measurements,iris_classes,iris_classnames)=load_iris_data()

classfiers_to_cv=[("kNN",knn),("Naive Bayes",nb)]

for (c_label, classifer) in classfiers_to_cv :

    print
    print "---> %s <---" % c_label

    best_k=0
    best_cv_a=0
    foldsizes=[2,3,5,10,15,30,50,75,150] 
    #These fold sizes are evenly divisible into the dataset intentionally

    for k_f in foldsizes :
       cv_a = cross_validate(iris_petal_measurements, iris_classes, classifer, k_fold=k_f)
       if cv_a >  best_cv_a :
            best_cv_a=cv_a
            best_k=k_f

       print "foldsize <<%s>> :: test accuracy <<%s>>" % (k_f, cv_a)


    print "Highest Accuracy: fold <<%s>> :: <<%s>>" % (best_k, best_cv_a)
