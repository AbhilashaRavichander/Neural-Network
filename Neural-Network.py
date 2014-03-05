'''
@Author : Abhilasha Ravichander
@Since  : 4th March, 2014, 4:00 PM
@About  : Simulates a neural network with one training tuple having inputs 1.0, 0.25 and -0.5 and with outputs 1.0 and -1.0. All values scaled 		  to range [0.0,1.0]
'''
import random,math

class Node:
	def __init__(self,layer,bias,error,inp,out):
		self.layer = layer
		self.bias = bias
		self.error = error
		self.inp = inp
		self.out = out
'''nodes: Dictionary to represent each of the nodes, key is node number, value is Node object having [layer,bias,error,input,output] for that node 1-inputlayer 2-hidden layer 3-output layer'''
nodes = {}

w = {} #w: Dictionary- key is (start point,end point) value is weight
deltaweight = {}
inputs = [1.0,0.25,-0.5]
outputs = [1.0,-1.0]
#scaling: (val-min)/(max-min)
for i,j in enumerate(inputs):
	inputs[i] = (inputs[i]+1)/2
for i,j in enumerate(outputs):
	outputs[i] = (outputs[i]+1)/2

l = .9 # learning rate : observations,larger learning rates, more likely to miss. Smaller learning rate, more iterations
acceptable = 0.0001 #Terminating condition : difference in weights should be less than this

#Initialise with random values, backpropogation will take care of this later
def initialise_network():
	for i in range(1,4): #Input layer
		nodes[i] = Node(1,int(random.uniform(0.0,1.0)*100)/100.0,0,inputs[i-1],inputs[i-1])
		for j in range(4,6):
			w[(i,j)] = int(random.uniform(0.0,1.0)*100)/100.0
			deltaweight[(i,j)] = 1.0
	for j in range(4,6): #Hidden layer
		nodes[j] = Node(2,int(random.uniform(0.0,1.0)*100)/100.0,0,0,0)
		for k in range(6,8):
			w[(j,k)] = int(random.uniform(0.0,1.0)*100)/100.0
	for i in range(6,8): #Output layer
		nodes[i] = Node(3,int(random.uniform(0.0,1.0)*100)/100.0,0,0,outputs[i%2])
	

#Print the values for the network
def print_values():
	print("	Weights")
	for i in range(1,4):
		for j in range(4,6):
			print(str(i)+"--->"+str(j)+" : "+"{0:.2f}".format(round(w[i,j],2)) )
	for j in range(4,6):
		for k in range(6,8):
			print(str(j)+"--->"+str(k)+" : "+"{0:.2f}".format(round(w[j,k],2)) )
	print("	Biases")
	for i in range(1,8):
		print("Node : "+str(i)+"  bias: "+"{0:.2f}".format(round(nodes[i].bias,2)))


#Calculate inputs for each layer: Ij = Summation(wij*Oi) + biasj
def calculate_inputs_outputs():
	for j in range(4,8): #Hidden layer
		nodes[j].inp=0
		nodes[j].out=0
		if(j in range(4,6)):
			for k in range(1,4):
				nodes[j].inp+=w[(k,j)]*nodes[k].out #input to hidden layer equal to weighted sum of outputs from previous layer
		else:
			for k in range(4,6):
				nodes[j].inp+=w[(k,j)]*nodes[k].out #input to output layer equal to weighted sum of outputs from previous layer
		nodes[j].inp = nodes[j].inp+nodes[j].bias
		#ip = nodes[j].inp
		#nodes[j][4] = 2/(1+math.pow(math.e,-(nodes[j][3])))-1
		#nodes[j][4] = ip/math.pow(1+math.pow(ip,2),0.5)
		nodes[j].out = 1/(1+math.pow(math.e,-(nodes[j].inp))) #Sigmoid function to squash used as activation fn

#Calculate errors and backpropogate
def backpropogation():
	for j in range(6,8):
		observed_output = nodes[j].out
		expected_output = outputs[j%2]
		nodes[j].error = observed_output*(1-observed_output)*(expected_output-observed_output)
	for j in range(4,6):
		observed_output = nodes[j].out
		nodes[j].error = observed_output*(1-observed_output)
		nextlayererror = 0
		for k in range(6,8):
			nextlayererror+=w[(j,k)]*nodes[k].error
		nodes[j].error+=nextlayererror
	#All errors computed, now we must update the weights and biases
	for j in w:
		deltaw = l * nodes[j[1]].error * nodes[j[0]].out #deltawij = l * Errj * Oi
		deltaweight[j] = deltaw
		w[j] = w[j] +deltaw
	for j in range(1,8):
		deltabias = l*nodes[j].error
		nodes[j].bias+=deltabias

def isTrained():
	for i in deltaweight:
		if deltaweight[i]>acceptable:
			return False
	print("Terminating condition : All changes in weights are below the specified threshold of 0.0001 as shown : ")
	print(deltaweight)
	return True

def neuralnetwork():		
	iterationCount=0
	initialise_network() #Initialise with random weights and biases
	print("Initial :")
	print_values()
	print("--------")
	calculate_inputs_outputs() #Calculate inputs and outputs at each node
	while(not(isTrained())):
		iterationCount+=1
		#print(" iteration : "+str(iterationCount))
		backpropogation()	#Backpropogation algorithm applied
		calculate_inputs_outputs() 
		#for i in nodes:
		#	print(str(i)+" output :"+str(nodes[i].out))
	print("--------")
	print("number of iterations : "+str(iterationCount))
	print("--------")
	print("Final :")
	print_values()

neuralnetwork()
	
