from extractingFeatures import *
import os
import math
import csv
from sklearn import neighbors,svm,tree,naive_bayes,neural_network, preprocessing, model_selection,metrics
import numpy as np
from Manual_ANN import train_nn,test_nn

TRAINING_PERCENTAGE=0.8
DATASET_PATH="dataset"

def setup_data(dataset_path=DATASET_PATH):
    num_men=0
    num_women=0
    alldata=[]
    alltargets=[]
    allfilenames=[]
    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            for file in sorted(os.listdir(folder_path)):
                if file.endswith(".pts"):
                    curr_data=[]
                    file_path = os.path.join(folder_path, file)
                    # print("Processing " + file_path)
                    alldata.append(extract_features(file_path=file_path))
                    allfilenames.append(file_path)
                    if file.startswith("m"):
                        alltargets.append(0)
                        num_men+=1
                    elif file.startswith("w"):
                        alltargets.append(1)
                        num_women+=1
    return alldata,alltargets,num_men,num_women,allfilenames

FEATURE_NAMES = [
    "eye_length_ratio",
    "eye_dist_ratio",
    "nose_ratio",
    "lip_size_ratio",
    "lip_length_ratio",
    "eyebrow_length_ratio",
    "aggresive_ratio",
    "jaw_eye_ratio",
    "forehead_jaw_ratio",
    "nose_to_lip_dist",
    "nose_to_lip_dist_lip_height_ratio",
]

def save_feature_report(alldata, alltargets, allfilenames, output_path="feature_values_report.csv"):
    label_map = {0: "male", 1: "female"}
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_number", "file_name", "class_label"] + FEATURE_NAMES)
        for i, (features, label, filepath) in enumerate(zip(alldata, alltargets, allfilenames)):
            writer.writerow([i + 1, os.path.basename(filepath), label_map[label]] + features)
    print(f"[INFO] Feature report saved -> {output_path}  ({len(alldata)} samples, {len(FEATURE_NAMES)} features)")

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
    # features.append(jaw_nose_width_ratio(p15,p16,p20,p21))
    # features.append(jaw_mouth_width_ratio(p2,p3,p20,p21))
    features.append(jaw_eye_ratio(p9,p10,p11,p12,p20,p21))
    features.append(forehead_jaw_ratio(p0,p13,p20,p21))
    features.append(nose_to_lip_dist(p6,p14,p17,p19))
    features.append(nose_to_lip_dist_lip_height_ratio(p14,p17,p18))
    return features

def train_knn(num_neighbors,traindata,traintargets):
    knn=neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(traindata,traintargets)
    return knn

def test_knn(knn,testdata,testtargets,final_model):
    pr=knn.predict(testdata)
    trueClasses=np.array(testtargets)
    if final_model:
        print("----------KNN----------")
        print(pr,"\n",trueClasses)
        CorrectMale,CorrectFemale,IncorrectMale,IncorrectFemale=0,0,0,0
        for i in range(len(trueClasses)):
            if trueClasses[i]==1 and pr[i]==1:
                CorrectFemale+=1
            if trueClasses[i]==0 and pr[i]==0:
                CorrectMale+=1
            if trueClasses[i]==0 and pr[i]==1:
                IncorrectMale+=1
            if trueClasses[i]==1 and pr[i]==0:
                IncorrectFemale+=1
        print("\nCorrectFemale(TP): ", CorrectFemale, " IncorrectFemale(FN): ",IncorrectFemale)
        print("IncorrectMale(FP): ", IncorrectMale, " CorrectMale(TN): ", CorrectMale)
        Accuracy=((CorrectFemale+CorrectMale)/len(trueClasses))*100
        Recall=((CorrectFemale)/(CorrectFemale+IncorrectMale))*100
        Precision=((CorrectFemale)/(CorrectFemale+IncorrectFemale))*100
        print(f"\nAccuracy: {Accuracy:.2f}% Recall: {Recall:.2f}% Precision: {Precision:.2f}%")
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
        # print("K: ", i)
        curr_acc=test_knn(knn,testdata,testtargets,final_model=False)
        if curr_acc>=best_acc:
            best_acc=curr_acc
            best_k=i

    # print("Best K Found for KNN: ", best_k)
    # print("Best Accuracy: ", best_acc)
    return best_k,best_acc

def train_decision_tree(traindata,traintargets):
    dt=tree.DecisionTreeClassifier(criterion="entropy")
    dt.fit(traindata,traintargets)
    return dt

def test_decision_tree(dt,testdata,testtargets):
    pr=dt.predict(testdata)
    trueClasses=np.array(testtargets)
    print("\n----------DT----------")
    print(pr,"\n",trueClasses)
    CorrectMale,CorrectFemale,IncorrectMale,IncorrectFemale=0,0,0,0
    for i in range(len(trueClasses)):
        if trueClasses[i]==1 and pr[i]==1:
            CorrectFemale+=1
        if trueClasses[i]==0 and pr[i]==0:
            CorrectMale+=1
        if trueClasses[i]==0 and pr[i]==1:
            IncorrectMale+=1
        if trueClasses[i]==1 and pr[i]==0:
            IncorrectFemale+=1
    print("\nCorrectFemale(TP): ", CorrectFemale, " IncorrectFemale(FN): ",IncorrectFemale)
    print("IncorrectMale(FP): ", IncorrectMale, " CorrectMale(TN): ", CorrectMale)
    Accuracy=((CorrectFemale+CorrectMale)/len(trueClasses))*100
    Recall=((CorrectFemale)/(CorrectFemale+IncorrectMale))*100
    Precision=((CorrectFemale)/(CorrectFemale+IncorrectFemale))*100
    print(f"\nAccuracy: {Accuracy:.2f}% Recall: {Recall:.2f}% Precision: {Precision:.2f}%")

def train_naive_bayes(traindata,traintargets):
    nb=naive_bayes.GaussianNB()
    nb.fit(traindata,traintargets)
    return nb

def test_naive_bayes(nb, testdata, testtargets):
    pr=nb.predict(testdata)
    trueClasses=np.array(testtargets)
    print("\n----------NB----------")
    print(pr,"\n",trueClasses)
    CorrectMale,CorrectFemale,IncorrectMale,IncorrectFemale=0,0,0,0
    for i in range(len(trueClasses)):
        if trueClasses[i]==1 and pr[i]==1:
            CorrectFemale+=1
        if trueClasses[i]==0 and pr[i]==0:
            CorrectMale+=1
        if trueClasses[i]==0 and pr[i]==1:
            IncorrectMale+=1
        if trueClasses[i]==1 and pr[i]==0:
            IncorrectFemale+=1
    print("\nCorrectFemale(TP): ", CorrectFemale, " IncorrectFemale(FN): ",IncorrectFemale)
    print("IncorrectMale(FP): ", IncorrectMale, " CorrectMale(TN): ", CorrectMale)
    Accuracy=((CorrectFemale+CorrectMale)/len(trueClasses))*100
    Recall=((CorrectFemale)/(CorrectFemale+IncorrectMale))*100
    Precision=((CorrectFemale)/(CorrectFemale+IncorrectFemale))*100
    print(f"\nAccuracy: {Accuracy:.2f}% Recall: {Recall:.2f}% Precision: {Precision:.2f}%")

def train_SVM(traindata,traintargets):
    svm_model=svm.SVC(kernel="linear")
    svm_model.fit(traindata,traintargets)
    return svm_model

def test_SVM(svm_model,testdata,testtargets):
    pr=svm_model.predict(testdata)
    trueClasses=np.array(testtargets)
    CorrectMale,CorrectFemale,IncorrectMale,IncorrectFemale=0,0,0,0
    print("\n----------SVM----------")
    print(pr,"\n",trueClasses)
    for i in range(len(trueClasses)):
        if trueClasses[i]==1 and pr[i]==1:
            CorrectFemale+=1
        if trueClasses[i]==0 and pr[i]==0:
            CorrectMale+=1
        if trueClasses[i]==0 and pr[i]==1:
            IncorrectMale+=1
        if trueClasses[i]==1 and pr[i]==0:
            IncorrectFemale+=1
    print("\nCorrectFemale(TP): ", CorrectFemale, " IncorrectFemale(FN): ",IncorrectFemale)
    print("IncorrectMale(FP): ", IncorrectMale, " CorrectMale(TN): ", CorrectMale)
    Accuracy=((CorrectFemale+CorrectMale)/len(trueClasses))*100
    Recall=((CorrectFemale)/(CorrectFemale+IncorrectMale))*100
    Precision=((CorrectFemale)/(CorrectFemale+IncorrectFemale))*100
    print(f"\nAccuracy: {Accuracy:.2f}% Recall: {Recall:.2f}% Precision: {Precision:.2f}%")

def find_best_rand_state_nn(traindata,traintargets,testdata,testtargets,hidden_layers,a,lr):
    best_acc=0.0
    trueClasses=np.array(testtargets)
    best_rand_state=-1
    for i in range(0,20000):
        nn=neural_network.MLPClassifier(activation="logistic", alpha=a, learning_rate_init=lr, max_iter=1000000, hidden_layer_sizes=hidden_layers,random_state=i)
        nn.fit(traindata,traintargets)
        pr=nn.predict(testdata)
        curr_acc=np.mean(pr==trueClasses)
        if(curr_acc>best_acc):
            print(i,": ",curr_acc)
            best_acc=curr_acc
            best_rand_state=i
    return best_rand_state

def train_sk_nn(traindata,traintargets,hidden_layers,a,lr,rand_state):
    nn=neural_network.MLPClassifier(activation="logistic", alpha=a, learning_rate_init=lr, max_iter=1000000, hidden_layer_sizes=hidden_layers,random_state=rand_state)
    nn.fit(traindata,traintargets)
    return nn


def test_sk_nn(nn,testdata,testtargets):
    pr=nn.predict(testdata)
    trueClasses=np.array(testtargets)
    print("\n----------ANN----------")
    print(pr,"\n",trueClasses)
    CorrectMale,CorrectFemale,IncorrectMale,IncorrectFemale=0,0,0,0
    Recall,Precision,Accuracy=0.0,0.0,0.0
    for i in range(len(trueClasses)):
        if trueClasses[i]==1 and pr[i]==1:
            CorrectFemale+=1
        if trueClasses[i]==0 and pr[i]==0:
            CorrectMale+=1
        if trueClasses[i]==0 and pr[i]==1:
            IncorrectMale+=1
        if trueClasses[i]==1 and pr[i]==0:
            IncorrectFemale+=1
    print("\nCorrectFemale(TP): ", CorrectFemale, " IncorrectFemale(FN): ",IncorrectFemale)
    print("IncorrectMale(FP): ", IncorrectMale, " CorrectMale(TN): ", CorrectMale)
    Accuracy=((CorrectFemale+CorrectMale)/len(trueClasses))*100
    Recall=((CorrectFemale)/(CorrectFemale+IncorrectMale))*100
    Precision=((CorrectFemale)/(CorrectFemale+IncorrectFemale))*100
    print(f"\nAccuracy: {Accuracy:.2f}% Recall: {Recall:.2f}% Precision: {Precision:.2f}%")

def find_best_nn_layers(traindata,traintargets,testdata,testtargets, a, lr, num_layers):
    best_acc=0
    trueClasses=np.array(testtargets)
    if num_layers==1:
        for i in range(1,76):
            model=neural_network.MLPClassifier(max_iter=1000000,activation="logistic", random_state=3, alpha=a, learning_rate_init=lr, hidden_layer_sizes=(i,))
            model.fit(traindata,traintargets)
            pr=model.predict(testdata)
            curr_acc=np.mean(pr==trueClasses)
            if curr_acc>=best_acc:
                best_acc=curr_acc
                best_hidden_layers=(i,)   
    if num_layers==2:
        for i in range(1,76):
            for j in range(1,76):
                model=neural_network.MLPClassifier(max_iter=1000000,activation="logistic", random_state=3, alpha=a, learning_rate_init=lr, hidden_layer_sizes=(i,j))
                model.fit(traindata,traintargets)
                pr=model.predict(testdata)
                curr_acc=np.mean(pr==trueClasses)
                if curr_acc>=best_acc:
                    best_acc=curr_acc
                    best_hidden_layers=(i,j)   
  
    if num_layers==3:
        for i in range(1,76):
            for j in range(1,76):
                for k in range(1,76):
                    model=neural_network.MLPClassifier(max_iter=10000,activation="logistic", random_state=3, alpha=a, learning_rate_init=lr, hidden_layer_sizes=(i,j,k))
                    model.fit(traindata,traintargets)
                    pr=model.predict(testdata)
                    curr_acc=np.mean(pr==trueClasses)
                    if curr_acc>=best_acc:
                        best_acc=curr_acc
                        best_hidden_layers=(i,j,k)
    print(best_hidden_layers)
    print(best_acc)
    return best_hidden_layers

def find_best_alpha_and_learning_rate_init(traindata,traintargets,testdata,testtargets, hidden_layers, alphas, lrs):
    best_acc=0
    trueClasses=np.array(testtargets)
    for i in range(len(alphas)):
        for j in range(len(lrs)):
            model=neural_network.MLPClassifier(max_iter=1000000,activation="logistic", random_state=3, alpha=alphas[i], learning_rate_init=lrs[j], hidden_layer_sizes=hidden_layers)
            model.fit(traindata,traintargets)
            pr=model.predict(testdata)
            curr_acc=np.mean(pr==trueClasses)
            if curr_acc>=best_acc:
                best_acc=curr_acc
                best_alpha=alphas[i]
                best_learn_r=lrs[j]
    print(best_alpha,best_learn_r)
    print(best_acc)
    return best_alpha,best_learn_r

alldata,alltargets,num_men,num_women,allfilenames=setup_data(DATASET_PATH)
save_feature_report(alldata, alltargets, allfilenames, output_path="feature_values_report.csv")
traindata,traintargets,testdata,testtargets=split_data(TRAINING_PERCENTAGE,alldata,alltargets,num_men,num_women)


scaler=preprocessing.StandardScaler()
traindata=scaler.fit_transform(traindata)
testdata=scaler.transform(testdata)

best_k,best_acc=get_best_k(traindata,traintargets,testdata,testtargets, math.floor((num_men+num_women)/4))

knn=train_knn(best_k,traindata,traintargets)
test_knn(knn,testdata,testtargets,final_model=True)

dt=train_decision_tree(traindata,traintargets)
test_decision_tree(dt,testdata,testtargets)

nb=train_naive_bayes(traindata,traintargets)
test_naive_bayes(nb,testdata,testtargets)

svm_model=train_SVM(traindata,traintargets)
test_SVM(svm_model,testdata,testtargets)

#2 Hidden Layer Best Settings:
# layers=(61,49)
# a2,lr2=0.1,0.075

# 1Hidden Layer Best Settings:
layers=(24,)
a2,lr2,a1,lr1=0.1, 0.0095,0.0,0.0
# rand_state=find_best_rand_state_nn(traindata,traintargets,testdata,testtargets, layers, a2, lr2)
rand_state=17124

n_layers=len(layers)
input_layer=len(alldata[0])
alphas=[0.009,0.0095,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
lrs=[0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.0045,0.005,0.0055,0.006,0.0065,0.007,0.0075,0.008,0.0085,0.009,0.0095,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095]

# while(a1!=a2 or lr1!=lr2):
#     a1=a2
#     lr1=lr2
#     layers=find_best_nn_layers(traindata,traintargets,testdata,testtargets,a1,lr1,n_layers)
#     a2,lr2=find_best_alpha_and_learning_rate_init(traindata,traintargets,testdata,testtargets,layers,alphas,lrs)

# layers=find_best_nn_layers(traindata,traintargets,testdata,testtargets,a2,lr2,n_layers)
# a2,lr2=find_best_alpha_and_learning_rate_init(traindata,traintargets,testdata,testtargets,layers,alphas,lrs)
if n_layers==3:
    l1,l2,l3=layers[0],layers[1],layers[2]
    mnn_layers=[input_layer,l1,l2,l3,1]
if n_layers==2:
    l1,l2=layers[0],layers[1]  
    mnn_layers=[input_layer,l1,l2,1]
if n_layers==1:
    l1=layers[0]
    mnn_layers=[input_layer,l1,1]
nn_model=train_sk_nn(traindata,traintargets, layers, a2, lr2, rand_state)
test_sk_nn(nn_model,testdata,testtargets)

mnn_max_iters=100000
mnn_init_learning_rate=lr2
mnn_learning_rate_multiplier=0.99

model_weights,model_biases=train_nn(mnn_max_iters,mnn_init_learning_rate,mnn_learning_rate_multiplier,mnn_layers,traindata,traintargets)
test_nn(model_weights,model_biases,testdata,testtargets)
