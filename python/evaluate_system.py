from biometric_system import *

#ASSUMPTION: All important classes are first k classes of all_classes

num_query = 2000
fraud_ratio = 0.1
num_day = 100 

num_query_total = num_query*num_day
tp = fp = tn = fn = 0

biometricSystem = BiometricSystem(database=initial_database, model=resnet, vgg_dataset=dataset)

for i in range(num_day):
    fraud = np.random.rand(num_query) < fraud_ratio
    labels = np.random.choice(imp_classes, num_query)
    for i in np.where(fraud)[0]:
        label = labels[i]
        newlabel = np.random.choice(all_classes)
        while(newlabel == label):
            newlabel = np.random.choice(all_classes)
        labels[i] = newlabel
    query_ids = [np.random.choice(dataset.class_to_instances[label]) for label in labels]
    query_ids = np.array(query_ids)
    pred = biometricSystem.checkfaces(query_ids)
    
    _tp = np.logical_and(pred == labels, pred >= 0)
    _fp = np.logical_and(pred != labels, pred >= 0)
    _tn = np.logical_and(labels > len(imp_classes)-1, pred < 0)
    _fn = np.logical_and(labels <= len(imp_classes)-1, pred < 0)
    tp += np.count_nonzero(_tp)
    fp += np.count_nonzero(_fp)
    tn += np.count_nonzero(_tn)
    fn += np.count_nonzero(_fn)
    
print(f"tp = {tp/num_query_total}%")
print(f"fp = {fp/num_query_total}%")
print(f"tn = {tn/num_query_total}%")
print(f"fn = {fn/num_query_total}%")

