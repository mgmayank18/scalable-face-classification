from biometric_system import *
import time

#ASSUMPTION: All important classes are first k classes of all_classes

num_query = 2000
fraud_ratio = 0.1
num_day = 2 

num_query_total = num_query*num_day
tp = fp = tn = fn = 0

biometricSystem = BiometricSystem(database=initial_database, model=resnet, vgg_dataset=dataset)

time_taken = []
for i in range(num_day):
    fraud = np.random.rand(num_query) < fraud_ratio
    labels = np.random.choice(imp_classes, num_query)
    labels[fraud] = np.random.choice(fraud_classes, len(labels[fraud]))

    query_ids = [np.random.choice(dataset.class_to_instances[label]) for label in labels]
    query_ids = np.array(query_ids)
    t = time.process_time()
    pred = biometricSystem.checkfaces(query_ids)
    elapsed_time = time.process_time() - t
    time_taken.append(elapsed_time)
    
    _tp = np.logical_and(pred == labels, pred >= 0)
    _fp = np.logical_and(pred != labels, pred >= 0)
    _tn = np.logical_and(labels > len(imp_classes)-1, pred < 0)
    _fn = np.logical_and(labels <= len(imp_classes)-1, pred < 0)
    tp += np.count_nonzero(_tp)
    fp += np.count_nonzero(_fp)
    tn += np.count_nonzero(_tn)
    fn += np.count_nonzero(_fn)
    
print(f"Average Time taken per day = {np.array(time_taken).mean()}")
print(f"tp = {100*tp/num_query_total}%")
print(f"fp = {100*fp/num_query_total}%")
print(f"tn = {100*tn/num_query_total}%")
print(f"fn = {100*fn/num_query_total}%")

