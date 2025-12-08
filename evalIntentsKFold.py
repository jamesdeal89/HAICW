# ====== K-FOLD INTENT CLASSIFICATION TESTING ======
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import readIntentsCsv, stemVectorWeight, generateInvertedIndex
from search import searchIntent
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Run k-fold cross-validation for intent classification
def runKFoldEvaluation(intents, numFolds=10):
    
    allPrompts = []
    allIntents = []
    
    # intents is a list of [prompt, intent] pairs
    for entry in intents:
        prompt = entry[0]
        intent = entry[1]
        allPrompts.append(prompt)
        allIntents.append(intent)
    
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    
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
                predictedIntent = "FAILED"
            yTrue.append(trueIntent)
            yPred.append(predictedIntent)
        
        # Calculate metrics
        accuracy = accuracy_score(yTrue, yPred)
        precision = precision_score(yTrue, yPred, average='weighted', zero_division=0)
        recall = recall_score(yTrue, yPred, average='weighted', zero_division=0)
        f1 = f1_score(yTrue, yPred, average='weighted', zero_division=0)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        # Clean up pickle files
        for picklePattern in [f'testpickle*_fold{fold}.pickle']:
            for pf in glob.glob(picklePattern):
                os.remove(pf)
    
    return {
        'accuracy': np.array(accuracies),
        'precision': np.array(precisions),
        'recall': np.array(recalls),
        'f1': np.array(f1s)
    }

# Run the K fold and collate resulsts and visualise.
def evaluateKFold():
    print("K-Fold Cross-Validation Evaluation")
    
    # Load dataset
    intents = readIntentsCsv()
    print("Running 10-fold cross-validation...")
    
    # Run k-fold cross-validation
    results = runKFoldEvaluation(intents, numFolds=10)
    
    print("\nResults:")
    for metricName, values in results.items():
        print(f"\n{metricName}:")
        print("Mean accuracy:", np.mean(values))
        print("Std deviation:", np.std(values))
        print("Min accuracy:", np.min(values))
        print("Max accuracy:", np.max(values))
        
    # Visualisation
    print("\nGenerating box plot visualization...")
    
    plt.figure(figsize=(8, 6))
    plt.boxplot(results['accuracy'], labels=['Intent Classifier'])
    plt.ylabel('Accuracy')
    plt.title('10-Fold Cross-Validation Results')
    plt.grid(axis='y', alpha=0.3)
    
    # Save the plot
    plt.savefig('kfold_boxplot.png', dpi=300, bbox_inches='tight')
    print("Box plot saved as 'kfold_boxplot.png'")
    plt.close()

if __name__ == "__main__":
    evaluateKFold()
