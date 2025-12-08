# ====== EVALUATION OF INTENT CLASSIFICATION ======
import os
import glob
from preprocessing import readIntentsCsv, stemVectorWeight, generateInvertedIndex
from search import searchIntent
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluateIntents():
    print("Intent Classification Evaluation")
    
    # Load dataset
    intents = readIntentsCsv()
    
    # Split dataset into training and testing sets
    trainData, testData = train_test_split(intents, test_size=0.2, random_state=42)
    
    print(f"\nDataset size: {len(intents)}")
    print(f"Training set: {len(trainData)}")
    print(f"Test set: {len(testData)}")
    
    # Clean up any existing pickle files
    print("\nCleaning up existing pickle files...")
    for pickleFile in glob.glob('*.pickle'):
        os.remove(pickleFile)
    
    # Train the classifier
    XtrainTf, count, tfidf = stemVectorWeight(trainData, False, 'testpickle1.pickle', 'testpickle2.pickle', 'testpickle3.pickle')
    invIdx = generateInvertedIndex(count, XtrainTf, 'testpickle11.pickle')
    
    # Predict on the test set
    yTrue = []
    yPred = []
    
    for testPrompt, trueIntent in testData:
        result = searchIntent(invIdx, testPrompt, count, tfidf, trainData, skip_confirmation=True)
        
        if result:
            predictedIntent = trainData[result[0]][1]
        else:
            predictedIntent = "NONE"
        
        yTrue.append(trueIntent)
        yPred.append(predictedIntent)
    
    # Evaluate the classifier
    accuracy = accuracy_score(yTrue, yPred)
    precision, recall, f1, support = precision_recall_fscore_support(
        yTrue, yPred, average='weighted', zero_division=0
    )
    
    print("\nRESULTS")
    print("Accuracy: ", accuracy)
    print("Precision:", precision)
    print("Recall:   ", recall)
    print("F1-Score: ", f1)
    
    # Per-class metrics
    print("\nPer-class Metrics:")
    
    uniqueIntents = sorted(set(yTrue))
    precisionPerClass, recallPerClass, f1PerClass, supportPerClass = precision_recall_fscore_support(
        yTrue, yPred, labels=uniqueIntents, zero_division=0
    )
    
    print(f"{'Intent':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    for i, intent in enumerate(uniqueIntents):
        print(f"{intent:<15} {precisionPerClass[i]:<12.4f} {recallPerClass[i]:<12.4f} "
              f"{f1PerClass[i]:<12.4f} {supportPerClass[i]:<10}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(yTrue, yPred, labels=uniqueIntents)
    print(cm)
    
    # Clean up test pickle files
    print("\nCleaning up test pickle files...")
    for pickleFile in glob.glob('testpickle*.pickle'):
        os.remove(pickleFile)

if __name__ == "__main__":
    evaluateIntents()
