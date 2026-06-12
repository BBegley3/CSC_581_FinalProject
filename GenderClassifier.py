from extractingFeatures import *
import os
import math
from sklearn import neighbors,svm,tree,naive_bayes
import numpy as np

TRAINING_PERCENTAGE=0.8
DATASET_PATH="dataset"

def setup_data(dataset_path=DATASET_PATH):
    num_men=0
    num_women=0
    alldata=[]
    alltargets=[]
    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            for file in sorted(os.listdir(folder_path)):
                if file.endswith(".pts"):
                    curr_data=[]
                    file_path = os.path.join(folder_path, file)
                    # print("Processing " + file_path)
                    alldata.append(extract_features(file_path=file_path))
                    if file.startswith("m"):
                        alltargets.append(0)
                        num_men+=1
                    elif file.startswith("w"):
                        alltargets.append(1)
                        num_women+=1
    return alldata,alltargets,num_men,num_women

def extract_features(file_path):
    features=[]
    with open(file_path, 'r') as file:
        lines = file.readlines()
        points = []
        for line in lines:
            if line.strip() and not line.startswith('version') and not line.startswith('n_points') and not line.startswith('format') and not line.startswith('{') and not line.startswith('}'):
                x, y = map(float, line.split())
                points.append((x, y))
    p0=points[0]
    p1=points[1]
    p2=points[2]
    p3=points[3]
    p4=points[4]
    p5=points[5]
    p6=points[6]
    p7=points[7]
    p8=points[8]
    p9=points[9]
    p10=points[10]
    p11=points[11]
    p12=points[12]
    p13=points[13]
    p14=points[14]
    p15=points[15]
    p16=points[16]
    p17=points[17]
    p18=points[18]
    p19=points[19]
    p20=points[20]
    p21=points[21]
    features.append(eye_length_ratio(p8,p9,p10,p11,p12,p13))
    features.append(eye_dist_ratio(p0,p1,p8,p13))
    features.append(nose_ratio(p15,p16,p20,p21))
    features.append(lip_size_ratio(p2,p3,p17,p18))
    features.append(lip_length_ratio(p2,p3,p20,p21))
    features.append(eyebrow_length_ratio(p4,p5,p6,p7,p8,p13))
    features.append(aggresive_ratio(p10,p19,p20,p21))
    return features

def train_knn(num_neighbors,traindata,traintargets):
    knn=neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(traindata,traintargets)
    return knn

def test_knn(knn,testdata,testtargets):
    pr=knn.predict(testdata)
    trueClasses=np.array(testtargets)
    print("KNN Accuracy: ", np.mean(pr==trueClasses))
    return np.mean(pr==trueClasses)

def classify_data(model,file_path):
    data=[extract_features(file_path)]
    pr=model.predict(data)
    return pr

def split_data(training_percentage,alldata,alltargets,num_men,num_women):
    MEN_TRAINING_BREAKPOINT=math.floor(num_men*training_percentage)
    WOMEN_TRAINING_BREAKPOINT=num_men+math.floor(num_women*training_percentage)
    DATASET_ENDPOINT=num_men+num_women

    traindata=[]
    traintargets=[]
    testdata=[]
    testtargets=[]

    for i in range(0,MEN_TRAINING_BREAKPOINT):
        traindata.append(alldata[i])
        traintargets.append(alltargets[i])

    for i in range(MEN_TRAINING_BREAKPOINT,num_men):
        testdata.append(alldata[i])
        testtargets.append(alltargets[i])

    for i in range(num_men,WOMEN_TRAINING_BREAKPOINT):
        traindata.append(alldata[i])
        traintargets.append(alltargets[i])

    for i in range(WOMEN_TRAINING_BREAKPOINT,DATASET_ENDPOINT):
        testdata.append(alldata[i])
        testtargets.append(alltargets[i])
    return traindata,traintargets,testdata,testtargets

def get_best_k(traindata,traintargets,testdata,testtargets, max_k):
    best_acc=0
    best_k=0

    for i in range(1,max_k+1):
        knn=train_knn(i,traindata,traintargets)
        print("K: ", i)
        curr_acc=test_knn(knn,testdata,testtargets)
        if curr_acc>=best_acc:
            best_acc=curr_acc
            best_k=i

    print("Best K: ", best_k)
    print("Best Accuracy: ", best_acc)
    return best_k,best_acc

def train_decision_tree(traindata,traintargets):
    dt=tree.DecisionTreeClassifier(criterion="entropy")
    dt.fit(traindata,traintargets)
    return dt

def test_decision_tree(dt,testdata,testtargets):
    pr=dt.predict(testdata)
    trueClasses=np.array(testtargets)
    print("DT Accuracy: ", np.mean(pr==trueClasses))
    return np.mean(pr==trueClasses)

def train_naive_bayes(traindata,traintargets):
    nb=naive_bayes.GaussianNB()
    nb.fit(traindata,traintargets)
    return nb

def test_naive_bayes(nb, testdata, testtargets):
    pr=nb.predict(testdata)
    trueClasses=np.array(testtargets)
    print("NB Accuracy: ", np.mean(pr==trueClasses))
    return np.mean(pr==trueClasses)

def train_SVM(traindata,traintargets):
    svm_model=svm.SVC()
    svm_model.fit(traindata,traintargets)
    return svm_model

def test_SVM(svm_model,testdata,testtargets):
    pr=svm_model.predict(testdata)
    trueClasses=np.array(testtargets)
    print("SVM Accuracy: ", np.mean(pr==trueClasses))
    return np.mean(pr==trueClasses)

alldata,alltargets,num_men,num_women=setup_data(DATASET_PATH)
traindata,traintargets,testdata,testtargets=split_data(TRAINING_PERCENTAGE,alldata,alltargets,num_men,num_women)
best_k,best_acc=get_best_k(traindata,traintargets,testdata,testtargets, math.floor(math.sqrt(num_men+num_women)))

knn=train_knn(best_k,traindata,traintargets)
knn_acc=test_knn(knn,testdata,testtargets)

dt=train_decision_tree(traindata,traintargets)
dt_acc=test_decision_tree(dt,testdata,testtargets)

nb=train_naive_bayes(traindata,traintargets)
nb_acc=test_naive_bayes(nb,testdata,testtargets)

svm_model=train_SVM(traindata,traintargets)
svm_acc=test_SVM(svm_model,testdata,testtargets)
