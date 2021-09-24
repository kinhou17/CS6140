# Naive Bayes Classifier Assignment
# Colin Nordquist

from glob import glob
import math
import os

spam_filepath = glob("/Users/colinnordquist/Downloads/HamSpam/spam/*")
ham_filepath = glob("/Users/colinnordquist/Downloads/HamSpam/ham/*")
test_filepath = glob("/Users/colinnordquist/Downloads/HamSpam/test/*")
spam_file_count = 0
ham_file_count = 0
total_word_count = 0
spam_words = {}
ham_words = {}

# Load data into dictionaries
for file in spam_filepath:
    spam_file_count += 1
    with open(file, "r") as curr_file:
        for line in curr_file:
            line = line.strip()
            total_word_count += 1
            if line in spam_words:
                spam_words[line] += 1
            else:
                spam_words[line] = 1

for file in ham_filepath:
    ham_file_count += 1
    with open(file, "r") as curr_file:
        for line in curr_file:
            line = line.strip()
            total_word_count += 1
            if line in ham_words:
                ham_words[line] += 1
            else:
                ham_words[line] = 1

total_file_count = ham_file_count + spam_file_count

# hyper-parameters
alpha = .005
vocab = 200000

# Smoothing
for key in spam_words:
    spam_words[key] += alpha

for key in ham_words:
    ham_words[key] += alpha

spam_prob = spam_file_count / (spam_file_count + ham_file_count)

# Classification
def classify(file):

    spam_words_prob = 0
    ham_words_prob = 0

    spam_final_prob = 0
    ham_final_prob = 0

    with open(file, "r") as f:
        for line in f:
            if (line.strip() in spam_words):
                word_probability = spam_words[line.strip()]/((alpha * vocab) + len(spam_words))
            else:
                word_probability = alpha/((alpha * vocab) + len(spam_words))
            spam_words_prob += math.log(word_probability)
    spam_final_prob = math.log(spam_prob) + spam_words_prob

    with open(file, "r") as f:
        for line in f:
            if (line.strip() in ham_words):
                word_probability = ham_words[line.strip()]/((alpha * vocab) + len(ham_words))
            else:
                word_probability = alpha/((alpha * vocab) + len(ham_words))
            ham_words_prob += math.log(word_probability)
    ham_final_prob = math.log(1 - spam_prob) + ham_words_prob

    if ham_final_prob >= spam_final_prob:
        return "HAM"
    else:
        return "SPAM"

# Run through test set for results
test_results = {}
test_results["TP"] = 0
test_results["TN"] = 0
test_results["FP"] = 0
test_results["FN"] = 0

spam_expected = {}
with open("/Users/colinnordquist/Downloads/HamSpam/truthfile", "r") as file:
    for line in file:
        spam_expected[line.strip()] = 0

for file in test_filepath:  
    prediction = classify(file)
    filename = os.path.basename(file).split(".")[0]
    
    if prediction == "SPAM":
        if filename in spam_expected:
            test_results["TP"] += 1
        else:
            test_results["FP"] += 1  

    if prediction == "HAM":
        if filename in spam_expected:
            test_results["FN"] += 1
        else:
            test_results["TN"] += 1  

TN = test_results["TN"]
TP = test_results["TP"]
FP = test_results["FP"]
FN = test_results["FN"]

accuracy = (TN + TP) / (TN + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f_score = (2 * precision * recall) / (precision + recall)

print("TestFile results:\n")
print("Alpha used: " + str(alpha))
print("Vocabulary size used: " + str(vocab))
print("\nTP:" + str(TP))
print("FP:" + str(FP))
print("FN:" + str(FN))
print("TN:" + str(TN))
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F Score: " + str(f_score))