import numpy as np

def sigmoid(input):
    return 1/(1+np.exp(-input))

def sigmoid_derivative(input):
    return input * (1-input)

def test_nn(weights,biases,testdata,testtargets):
    testdata=np.array(testdata)
    testtargets=np.array(testtargets)
    activations=forward_prop(testdata,weights,biases)
    predictions=activations[-1].flatten()
    predicted_classes=[1 if prediction >=0.5 else 0 for prediction in predictions]
    predicted_classes=np.array(predicted_classes)
    correct_predictions=np.sum(predicted_classes==testtargets.flatten())
    accuracy=((correct_predictions/len(testtargets))*100)
    return accuracy


def train_nn(iterations,init_learning_rate,learn_multiplier,layers,traindata,traintargets):
    weights=[]
    biases=[]
    traindata=np.array(traindata)
    learning_rate=init_learning_rate

    for i in range(len(layers)-1):
        w=np.random.uniform(-0.5,0.5,(layers[i],layers[i+1]))
        b=np.random.uniform(-0.5,0.5,(1,layers[i+1]))
        weights.append(w)
        biases.append(b)
    
    prev_weights=[w.copy() for w in weights]

    for iteration in range(iterations):
        activations=forward_prop(traindata,weights,biases)
        weights,biases=back_prop(activations,traintargets,weights,biases,learning_rate)
        converged=True
        for i in range(len(weights)):
            if(np.any(abs(weights[i]-prev_weights[i])>10**-8)):
                converged=False
        if(converged):
            break
        prev_weights=[w.copy() for w in weights]
        if (iteration % 1000==0):
            learning_rate*=learn_multiplier
    return weights,biases

def forward_prop(X, weights, biases):
    activations=[X]
    current_activation=X
    for(w,b) in zip(weights,biases):
        net_input=np.dot(current_activation, w)+b
        current_activation=sigmoid(net_input)
        activations.append(current_activation)
    return activations

def back_prop(activations,targets,weights,biases,learning_rate):
    weight_gradients=[None]*len(weights)
    bias_gradients=[None]*len(biases)

    output_activation=activations[-1]
    
    targets = np.array(targets).reshape(-1, 1)
    deltaKs=(targets-output_activation)*sigmoid_derivative(output_activation)

    for l in reversed(range(len(weights))):
        weight_gradients[l]=np.dot(activations[l].T,deltaKs)
        bias_gradients[l]=np.sum(deltaKs,axis=0,keepdims=True)

        if l>0:
            next_layer_err=np.dot(deltaKs, weights[l].T)
            previous_activation=activations[l]
            deltaKs=next_layer_err*sigmoid_derivative(previous_activation)
        
    for l in range(len(weights)):
        weights[l] += learning_rate * weight_gradients[l]
        biases[l] += learning_rate * bias_gradients[l]

    return weights,biases

