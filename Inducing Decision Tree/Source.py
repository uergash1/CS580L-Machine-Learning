from random import randint
import os.path
import numpy as np
import pandas as pd
import math
import copy 
import sys
import ast

class node:
	def __init__(self):
		self.is_leaf  = False
		self.classification = None
		self.left = None
		self.right = None
		self.possible_class = None
		self.tag = None
		self.index = None

class data:
    def __init__(self):
    	self.examples = []
    	self.attributes = []
    	#self.chosen_attributes_indexes = []
    	self.attributes_index = []

def read_data(dataset, filePath):
	f = open(filePath)
	original_file = f.read()
	rowsplit_data = original_file.splitlines()
	dataset.attributes = [rows.split(',') for rows in rowsplit_data].pop(0)

	dataset.examples = pd.read_csv(filePath)

	#not yet chosen attribute indexes 
	for i in range(len(dataset.attributes) - 1):
		dataset.attributes_index.append(i)

def run_decision_tree(dataset, heuristic):
	root = node()
	root = choose_attribute(dataset, heuristic)
	build_tree(dataset, root, heuristic)
	return root

def build_tree(dataset, root, heuristic):
	#right side of the root
	dataset1 = copy.deepcopy(dataset)
	dataset1.examples = dataset.examples[dataset.examples.iloc[:,root.index]==1]
	newNode = choose_attribute(dataset1, heuristic)
	root.right = newNode
	if newNode.is_leaf is False:
		ret = build_tree(dataset1, root.right, heuristic)

	#left side of the root
	dataset1 = copy.deepcopy(dataset)
	dataset1.examples = dataset.examples[dataset.examples.iloc[:,root.index]==0]
	newNode = choose_attribute(dataset1, heuristic)
	root.left = newNode
	if newNode.is_leaf is False:
		ret = build_tree(dataset1, root.left, heuristic)


def choose_attribute(dataset, heuristic): #choose best attribute
	newNode = node()
	bestIG = 0
	bestIGNodePosition = 0
	currentIG = 0
	indexPosition = 0
	numberOfAttributes = len(dataset.attributes_index)

	if heuristic == "first":
		result = get_entropy(dataset)
	else:
		result = get_variance(dataset)

	entropyBeforeSplit = result['result']

	if entropyBeforeSplit == 0:
		newNode.classification = result['classification']
		newNode.is_leaf = True
		newNode.tag = -1
		return newNode

	classColumn = dataset.examples.shape[1]
	posExamples = dataset.examples[dataset.examples.iloc[:,classColumn-1]==1].shape[0]
	negExamples = dataset.examples[dataset.examples.iloc[:,classColumn-1]==0].shape[0]

	if numberOfAttributes == 0:
		if posExamples > negExamples:
			newNode.classification = 1
		else:
			newNode.classification = 0

		newNode.tag = -1
		newNode.is_leaf = True
		return newNode
	
	if posExamples > negExamples:
		newNode.possible_class = 1
	else:
		newNode.possible_class = 0

	for i in range(numberOfAttributes):
		dataset1 = copy.deepcopy(dataset)
		dataset1.examples = dataset.examples[dataset.examples.iloc[:,dataset.attributes_index[i]]==0]

		if heuristic == "first":
			result = get_entropy(dataset1)
		else:
			result = get_variance(dataset1)

		entropyAfterSplit0 = result['result']

		dataset1 = copy.deepcopy(dataset)
		dataset1.examples = dataset.examples[dataset.examples.iloc[:,dataset.attributes_index[i]]==1]

		if heuristic == "first":
			result = get_entropy(dataset1)
		else:
			result = get_variance(dataset1)

		entropyAfterSplit1 = result['result']

		numberOfExamples = dataset.examples.shape[0]

		negExamples = dataset.examples[dataset.examples.iloc[:,dataset.attributes_index[i]]==0].shape[0]
		posExamples = dataset.examples[dataset.examples.iloc[:,dataset.attributes_index[i]]==1].shape[0]

		currentIG = entropyBeforeSplit - (negExamples / numberOfExamples) * entropyAfterSplit0 - (posExamples / numberOfExamples) * entropyAfterSplit1

		if currentIG > bestIG:
			bestIG = currentIG
			bestIGNodePosition = dataset.attributes_index[i]
			indexPosition = i

	#dataset.chosen_attributes_indexes.append(bestIGNodePosition)
	
	
	newNode.index = dataset.attributes_index.pop(indexPosition)
	newNode.tag = dataset.attributes[newNode.index]
	return newNode

def get_entropy(dataset):
	#if target_attr is None:
	classColumn = dataset.examples.shape[1]
	numberOfExamples = dataset.examples.shape[0]
	posExamples = dataset.examples[dataset.examples.iloc[:,classColumn-1]==1]
	negExamples = dataset.examples[dataset.examples.iloc[:,classColumn-1]==0]
	posExamples = posExamples.shape[0]
	negExamples = negExamples.shape[0]

	result = 0
	classification = 0
	if numberOfExamples == 0:
		return {'result':result,'classification':classification}
	pos = posExamples/numberOfExamples
	neg = negExamples/numberOfExamples
	if posExamples is not 0 and negExamples is not 0:
		result = -(pos * math.log(pos,2)) - (neg * math.log(neg,2))
		return {'result':result,'classification':classification}
	else:
		if posExamples > negExamples:
			classification = 1
		else:
			classification = 0
		return {'result':result,'classification':classification}

def get_variance(dataset):
	classColumn = dataset.examples.shape[1]
	numberOfExamples = dataset.examples.shape[0]
	posExamples = dataset.examples[dataset.examples.iloc[:,classColumn-1]==1].shape[0]
	negExamples = dataset.examples[dataset.examples.iloc[:,classColumn-1]==0].shape[0]

	result = 0
	classification = 0

	if numberOfExamples == 0:
		return {'result':result,'classification':classification}

	pos = posExamples/numberOfExamples
	neg = negExamples/numberOfExamples

	if posExamples is not 0 and negExamples is not 0:
		result = pos * neg
		return {'result':result,'classification':classification}
	else:
		if posExamples > negExamples:
			classification = 1
		else:
			classification = 0
		return {'result':result,'classification':classification}

def prune_tree(root, dataset, attributeIndexes):
	while (1):
		node = find_prune_node_second(root, attributeIndexes)

		if node is None:
			break

		valBeforePruning = get_accuracy(root, dataset.examples)

		#saving all features before deleting
		classification = node.classification
		left = node.left
		right = node.right
		possible_class = node.possible_class
		tag = node.tag
		index = node.index

		#making the node leaf node
		node.is_leaf = True
		node.classification = node.possible_class
		node.left = None
		node.right = None
		node.tag = -1
		node.index = None

		valAfterPruning	= get_accuracy(root, dataset.examples)

		if valAfterPruning >= valBeforePruning:
			attributeIndexes.pop(index)
		else:
			node.is_leaf = False
			node.classification = classification
			node.left = left
			node.right = right
			node.possible_class = possible_class
			node.tag = tag
			node.index = index
			break
	return root

def find_prune_node(root, attributeIndexes):
	for index in attributeIndexes:
		node = get_node(root, int(index))
		if node.left is not None and node.right is not None:
			if node.left.is_leaf is True and node.right.is_leaf is True:
				return node
	return None

def find_prune_node_second(root, attributeIndexes):
	while (1):
		attributeIndexes = np.asarray(attributeIndexes)
		index = randint(0, len(attributeIndexes)-1)
		node = get_node(root, attributeIndexes[index])
		if node is not None and node.left is not None and node.right is not None:
			if node.left.is_leaf is True and node.right.is_leaf is True:
				return node

def get_node(root, index):
	stack = []
	node = root
	while node!= None or len(stack)>0:
		if node != None:
			stack.append(node)
			node = node.left
		else:
			node = stack.pop()
			if node.index == index:
				return node
			node = node.right

def print_tree(root, level, file):
	if(root.left.is_leaf == True):
		file.write('|'*level+root.tag+" = 0 : " +str(root.left.classification)+'\n')
	else:
		file.write('|'*level+root.tag+" = 0"+'\n')
		print_tree(root.left,level+1,file)
        
	if(root.right.is_leaf == True):
		file.write('|'*level+root.tag+" = 1 : " +str(root.right.classification)+'\n')
        
	else:
		file.write('|'*level+root.tag+" = 1"+'\n')
		print_tree(root.right,level+1,file)

def is_match(root,test):
    currNode = root
    while(1):
        if currNode.is_leaf is True:
            break
        currIndex = int(currNode.index)
        valueOfCurrentIndex = test.iloc[currIndex]
        if(valueOfCurrentIndex == 1):
            currNode = currNode.right
        elif(valueOfCurrentIndex == 0):
            currNode = currNode.left
        else:
            assert False
    classVar = test.shape[0]
    classVar = classVar -1 
    if(currNode.classification == test.iloc[classVar]):
        return True
    else:
        return False

def get_accuracy(root, tests):
    testCasesRan = 0
    testCasesPassed = 0
    for index,test in tests.iterrows():
        result = is_match(root,test)
        testCasesRan = testCasesRan +1
        if(result == True):
            testCasesPassed +=1
    accuracy = testCasesPassed/testCasesRan
    return accuracy

def main():
	command_line_args = str(sys.argv)
	command_line_args = ast.literal_eval(command_line_args)

	if(len(command_line_args) < 6):
		print("Invalid command line parameters!")
		print("Usage: .\program <training-set> <validation-set> <test-set> <to-print> to-print:{yes,no} <prune> prune:{yes, no}")

	else:
		training_path = str(command_line_args[1])
		validation_path = str(command_line_args[2])
		test_path = str(command_line_args[3])
		to_print_tree = str(command_line_args[4].split(":")[1])
		to_prune_tree  = str(command_line_args[5].split(":")[1])

		if to_print_tree == 'yes' and to_prune_tree == 'yes':
			the_file = open('output.txt', 'a')

			####################################################
			# First Heuristic Not Pruned 
			####################################################

			#building a decision tree
			dataset = data()
			read_data(dataset, training_path)
			attributeIndexes = dataset.attributes_index
			root = run_decision_tree(dataset, "first")

			#calculate accuracy of decision tree on training
			dataset1 = data()
			read_data(dataset1, training_path)
			the_file.write('H1 NP Training: ' + str(get_accuracy(root, dataset1.examples)) + '\n')
						
			#calculate accuracy of decision tree on test data
			dataset2 = data()
			read_data(dataset2, validation_path)
			the_file.write('H1 NP Validation: ' + str(get_accuracy(root, dataset2.examples)) + '\n')

			#calculate accuracy of decision tree on validation data
			dataset3 = data()
			read_data(dataset3, test_path)
			the_file.write('H1 NP Test: ' + str(get_accuracy(root, dataset3.examples)) + '\n')

			####################################################
			# First Heuristic Pruned 
			####################################################

			#pruning DT
			dataset4 = data()
			read_data(dataset4, validation_path)
			root = prune_tree(root, dataset4, attributeIndexes)

			#calculate accuracy of decision tree on training data
			dataset5 = data()
			read_data(dataset5, training_path)
			the_file.write('H1 P Training: ' + str(get_accuracy(root, dataset5.examples)) + '\n')
						
			#calculate accuracy of decision tree on test data
			dataset6 = data()
			read_data(dataset6, validation_path)
			the_file.write('H1 P Validation: ' + str(get_accuracy(root, dataset6.examples)) + '\n')

			#calculate accuracy of decision tree on validation data
			dataset7 = data()
			read_data(dataset7, test_path)
			the_file.write('H1 P Test: ' + str(get_accuracy(root, dataset7.examples)) + '\n')

			#print tree
			print_tree(root, 0, the_file)

			####################################################
			# Second Heuristic Not Pruned 
			####################################################

			#building a decision tree
			dataset8 = data()
			read_data(dataset8, training_path)
			attributeIndexes1 = dataset8.attributes_index
			root = run_decision_tree(dataset8, "second")

			#calculate accuracy of decision tree on training
			dataset9 = data()
			read_data(dataset9, training_path)
			the_file.write('H2 NP Training: ' + str(get_accuracy(root, dataset9.examples)) + '\n')
						
			#calculate accuracy of decision tree on test data
			dataset10 = data()
			read_data(dataset10, validation_path)
			the_file.write('H2 NP Validation: ' + str(get_accuracy(root, dataset10.examples)) + '\n')

			#calculate accuracy of decision tree on validation data
			dataset11 = data()
			read_data(dataset11, test_path)
			the_file.write('H2 NP Test: ' + str(get_accuracy(root, dataset11.examples)) + '\n')

			####################################################
			# Second Heuristic Pruned 
			####################################################

			#pruning DT
			dataset12 = data()
			read_data(dataset12, validation_path)
			root = prune_tree(root, dataset12, attributeIndexes1)

			#calculate accuracy of decision tree on training data
			dataset13 = data()
			read_data(dataset13, training_path)
			the_file.write('H2 P Training: ' + str(get_accuracy(root, dataset13.examples)) + '\n')
						
			#calculate accuracy of decision tree on test data
			dataset14 = data()
			read_data(dataset14, validation_path)
			the_file.write('H2 P Validation: ' + str(get_accuracy(root, dataset14.examples)) + '\n')

			#calculate accuracy of decision tree on validation data
			dataset15 = data()
			read_data(dataset15, test_path)
			the_file.write('H2 P Test: ' + str(get_accuracy(root, dataset15.examples)) + '\n')

			#print tree
			print_tree(root, 0, the_file)
			
		elif to_print_tree == 'no' and to_prune_tree == 'yes':
			the_file = open('output.txt', 'a')
			####################################################
			# First Heuristic Not Pruned 
			####################################################

			#building a decision tree
			dataset = data()
			read_data(dataset, training_path)
			attributeIndexes = dataset.attributes_index
			root = run_decision_tree(dataset, "first")

			#calculate accuracy of decision tree on training
			dataset1 = data()
			read_data(dataset1, training_path)
			the_file.write('H1 NP Training: ' + str(get_accuracy(root, dataset1.examples)) + '\n')
						
			#calculate accuracy of decision tree on test data
			dataset2 = data()
			read_data(dataset2, validation_path)
			the_file.write('H1 NP Validation: ' + str(get_accuracy(root, dataset2.examples)) + '\n')

			#calculate accuracy of decision tree on validation data
			dataset3 = data()
			read_data(dataset3, test_path)
			the_file.write('H1 NP Test: ' + str(get_accuracy(root, dataset3.examples)) + '\n')

			####################################################
			# First Heuristic Pruned 
			####################################################

			#pruning DT
			dataset4 = data()
			read_data(dataset4, validation_path)
			root = prune_tree(root, dataset4, attributeIndexes)

			#calculate accuracy of decision tree on training data
			dataset5 = data()
			read_data(dataset5, training_path)
			the_file.write('H1 P Training: ' + str(get_accuracy(root, dataset5.examples)) + '\n')
						
			#calculate accuracy of decision tree on test data
			dataset6 = data()
			read_data(dataset6, validation_path)
			the_file.write('H1 P Validation: ' + str(get_accuracy(root, dataset6.examples)) + '\n')

			#calculate accuracy of decision tree on validation data
			dataset7 = data()
			read_data(dataset7, test_path)
			the_file.write('H1 P Test: ' + str(get_accuracy(root, dataset7.examples)) + '\n')

			####################################################
			# Second Heuristic Not Pruned 
			####################################################

			#building a decision tree
			dataset8 = data()
			read_data(dataset8, training_path)
			attributeIndexes1 = dataset8.attributes_index
			root = run_decision_tree(dataset8, "second")

			#calculate accuracy of decision tree on training
			dataset9 = data()
			read_data(dataset9, training_path)
			the_file.write('H2 NP Training: ' + str(get_accuracy(root, dataset9.examples)) + '\n')
						
			#calculate accuracy of decision tree on test data
			dataset10 = data()
			read_data(dataset10, validation_path)
			the_file.write('H2 NP Validation: ' + str(get_accuracy(root, dataset10.examples)) + '\n')

			#calculate accuracy of decision tree on validation data
			dataset11 = data()
			read_data(dataset11, test_path)
			the_file.write('H2 NP Test: ' + str(get_accuracy(root, dataset11.examples)) + '\n')

			####################################################
			# Second Heuristic Pruned 
			####################################################

			#pruning DT
			dataset12 = data()
			read_data(dataset12, validation_path)
			root = prune_tree(root, dataset12, attributeIndexes1)

			#calculate accuracy of decision tree on training data
			dataset13 = data()
			read_data(dataset13, training_path)
			the_file.write('H2 P Training: ' + str(get_accuracy(root, dataset13.examples)) + '\n')
						
			#calculate accuracy of decision tree on test data
			dataset14 = data()
			read_data(dataset14, validation_path)
			the_file.write('H2 P Validation: ' + str(get_accuracy(root, dataset14.examples)) + '\n')

			#calculate accuracy of decision tree on validation data
			dataset15 = data()
			read_data(dataset15, test_path)
			the_file.write('H2 P Test: ' + str(get_accuracy(root, dataset15.examples)) + '\n')

		elif to_print_tree == 'yes' and to_prune_tree == 'no':
			the_file = open('output.txt', 'a')
			####################################################
			# First Heuristic Not Pruned 
			####################################################

			#building a decision tree
			dataset = data()
			read_data(dataset, training_path)
			attributeIndexes = dataset.attributes_index
			root = run_decision_tree(dataset, "first")

			#calculate accuracy of decision tree on training
			dataset1 = data()
			read_data(dataset1, training_path)
			the_file.write('H1 NP Training: ' + str(get_accuracy(root, dataset1.examples)) + '\n')
						
			#calculate accuracy of decision tree on test data
			dataset2 = data()
			read_data(dataset2, validation_path)
			the_file.write('H1 NP Validation: ' + str(get_accuracy(root, dataset2.examples)) + '\n')

			#calculate accuracy of decision tree on validation data
			dataset3 = data()
			read_data(dataset3, test_path)
			the_file.write('H1 NP Test: ' + str(get_accuracy(root, dataset3.examples)) + '\n')

			#print tree
			print_tree(root, 0, the_file)

			####################################################
			# Second Heuristic Not Pruned 
			####################################################

			#building a decision tree
			dataset8 = data()
			read_data(dataset8, training_path)
			attributeIndexes1 = dataset8.attributes_index
			root = run_decision_tree(dataset8, "second")

			#calculate accuracy of decision tree on training
			dataset9 = data()
			read_data(dataset9, training_path)
			the_file.write('H2 NP Training: ' + str(get_accuracy(root, dataset9.examples)) + '\n')
						
			#calculate accuracy of decision tree on test data
			dataset10 = data()
			read_data(dataset10, validation_path)
			the_file.write('H2 NP Validation: ' + str(get_accuracy(root, dataset10.examples)) + '\n')

			#calculate accuracy of decision tree on validation data
			dataset11 = data()
			read_data(dataset11, test_path)
			the_file.write('H2 NP Test: ' + str(get_accuracy(root, dataset11.examples)) + '\n')

			#print tree
			print_tree(root, 0, the_file)

		elif to_print_tree == 'no' and to_prune_tree == 'no':
			the_file = open('output.txt', 'a')
			####################################################
			# First Heuristic Not Pruned 
			####################################################

			#building a decision tree
			dataset = data()
			read_data(dataset, training_path)
			attributeIndexes = dataset.attributes_index
			root = run_decision_tree(dataset, "first")

			#calculate accuracy of decision tree on training
			dataset1 = data()
			read_data(dataset1, training_path)
			the_file.write('H1 NP Training: ' + str(get_accuracy(root, dataset1.examples)) + '\n')
						
			#calculate accuracy of decision tree on test data
			dataset2 = data()
			read_data(dataset2, validation_path)
			the_file.write('H1 NP Validation: ' + str(get_accuracy(root, dataset2.examples)) + '\n')

			#calculate accuracy of decision tree on validation data
			dataset3 = data()
			read_data(dataset3, test_path)
			the_file.write('H1 NP Test: ' + str(get_accuracy(root, dataset3.examples)) + '\n')

			####################################################
			# Second Heuristic Not Pruned 
			####################################################

			#building a decision tree
			dataset8 = data()
			read_data(dataset8, training_path)
			attributeIndexes1 = dataset8.attributes_index
			root = run_decision_tree(dataset8, "second")

			#calculate accuracy of decision tree on training
			dataset9 = data()
			read_data(dataset9, training_path)
			the_file.write('H2 NP Training: ' + str(get_accuracy(root, dataset9.examples)) + '\n')
						
			#calculate accuracy of decision tree on test data
			dataset10 = data()
			read_data(dataset10, validation_path)
			the_file.write('H2 NP Validation: ' + str(get_accuracy(root, dataset10.examples)) + '\n')

			#calculate accuracy of decision tree on validation data
			dataset11 = data()
			read_data(dataset11, test_path)
			the_file.write('H2 NP Test: ' + str(get_accuracy(root, dataset11.examples)) + '\n')

if __name__ == "__main__":
    main()