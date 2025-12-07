import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import readIntentsCsv, stemVectorWeight, generateInvertedIndex
from search import searchIntent
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def runKFoldEvaluation(intents, numFolds=10):
    """Run k-fold cross-validation for intent classification"""
    
    allPrompts = []
    allIntents = []
    
    # intents is a list of [prompt, intent] pairs
    for entry in intents:
        prompt = entry[0]
        intent = entry[1]
        allPrompts.append(prompt)
        allIntents.append(intent)
    
    accuracies = []
    
    for fold in range(numFolds):
        # Split data with different random state each fold
        trainPrompts, testPrompts, trainIntents, testIntents = train_test_split(
            allPrompts, allIntents, test_size=0.2, random_state=fold
        )
        
        # Prepare training data
        trainData = [[prompt, intent] for prompt, intent in zip(trainPrompts, trainIntents)]
        
        # Train classifier
        pickleFile = f'testpickle_fold{fold}.pickle'
        XtrainTf, count, tfidf = stemVectorWeight(trainData, False, pickleFile, 
                                                   f'testpickle1_fold{fold}.pickle', 
                                                   f'testpickle2_fold{fold}.pickle')
        
        # Generate inverted index
        invIdx = generateInvertedIndex(count, XtrainTf, f'testpickle_inv_fold{fold}.pickle')
        
        # Evaluate on test set
        yTrue = []
        yPred = []
        
        for testPrompt, trueIntent in zip(testPrompts, testIntents):
            result = searchIntent(invIdx, testPrompt, count, tfidf, trainData, skip_confirmation=True)
            if result is not None:
                predictedDocId = result[0]
                predictedIntent = trainData[predictedDocId][1]
            else:
                predictedIntent = None
            yTrue.append(trueIntent)
            yPred.append(predictedIntent)
        
        # Calculate accuracy
        accuracy = accuracy_score(yTrue, yPred)
        accuracies.append(accuracy)
        
        # Clean up pickle files
        for picklePattern in [f'testpickle*_fold{fold}.pickle']:
            for pf in glob.glob(picklePattern):
                os.remove(pf)
    
    return np.array(accuracies)

def evaluateKFold():
    print("K-Fold Cross-Validation Evaluation")
    
    # Load dataset
    intents = readIntentsCsv()
    print("Running 10-fold cross-validation...")
    
    # Run k-fold cross-validation
    accuracies = runKFoldEvaluation(intents, numFolds=10)
    
    print("\nResults:")
    print("10-fold CV accuracies:", accuracies)
    print("Mean accuracy:", np.mean(accuracies))
    print("Std deviation:", np.std(accuracies))
    print("Min accuracy:", np.min(accuracies))
    print("Max accuracy:", np.max(accuracies))
    
    # Visualization - Simple box plot
    print("\nGenerating box plot visualization...")
    
    plt.figure(figsize=(8, 6))
    plt.boxplot(accuracies, labels=['Intent Classifier'])
    plt.ylabel('Accuracy')
    plt.title('10-Fold Cross-Validation Results')
    plt.grid(axis='y', alpha=0.3)
    
    # Save the plot
    plt.savefig('kfold_boxplot.png', dpi=300, bbox_inches='tight')
    print("Box plot saved as 'kfold_boxplot.png'")
    plt.close()

if __name__ == "__main__":
    evaluateKFold()
