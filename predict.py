import numpy as np
import csv
import sys
import pickle

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_nb.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""
def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, dtype = 'str', delimiter='\n')
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)
    return test_X, model


def predict_target_values(test_data, model):

    class_wise_frequency_dict, class_wise_denominators, prior_probabilities = model
    def compute_likelihood(test_X, c, class_wise_frequency_dict, class_wise_denominators):
    
        #TODO Complete the function implementation. Read the Question text for details
        vocab = [[word for word in class_wise_frequency_dict[words]]
            for words in class_wise_frequency_dict.keys()]
        vocalbulary = []
        for i in range(len(vocab)):
            vocalbulary += vocab[i]
        vocalbulary = list(set(vocalbulary))
        likelihood = 0
        for word in test_X.split():
            if word in vocalbulary:
                if word in class_wise_frequency_dict[c]:
                    likelihood += np.log((class_wise_frequency_dict[c][word] + 1)/\
                    class_wise_denominators[c])
                else:
                    likelihood -= np.log(class_wise_denominators[c])
            else:
                likelihood -= np.log(class_wise_denominators[c] + 1)
        return likelihood
  
    pred_Y = []

    def predict_class(test_X,class_wise_frequency_dict,\
        class_wise_denominators, prior_probabilities):
            predicted_class = {}
            for classes in class_wise_denominators.keys():
                predicted_class[classes] = np.log(prior_probabilities[classes])\
                + compute_likelihood(test_X, classes, class_wise_frequency_dict, class_wise_denominators)
            predicted_class = sorted(predicted_class.items(), key = lambda x: -x[1])
            return predicted_class[0][0]


    for test_X in test_data:
        pred_class = predict_class(test_X, class_wise_frequency_dict, class_wise_denominators,\
        prior_probabilities)
        print(pred_class)
        pred_Y.append(pred_class)
    pred_Y = np.array(pred_Y)
    return pred_Y    
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    # Load Model Parameters
    """
    Parameters of Naive Bayes include Laplace smoothing parameter, Prior probabilites of each class and values related to likelihood computation.
    """
    test_X, model = import_data_and_model(test_X_file_path, 'weights.pkl')
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_nb.csv")    


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_nb.csv") 