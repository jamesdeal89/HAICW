import os, glob
from preprocessing import readIntentsCsv, stemVectorWeight, generateInvertedIndex
from search import searchIntent
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

def evaluate_intents():
    print("INTENT CLASSIFICATION EVALUATION")
    
    # Load data
    intents = readIntentsCsv()
    
    # Split into train/test (80/20)
    train_data, test_data = train_test_split(intents, test_size=0.2, random_state=42)
    
    print(f"\nDataset size: {len(intents)}")
    print(f"Training set: {len(train_data)}")
    print(f"Test set: {len(test_data)}")
    
    # Clean up any existing pickle files
    print("\nCleaning up existing pickle files...")
    for pickle_file in glob.glob('*.pickle'):
        os.remove(pickle_file)
    
    # Train on training data
    XtrainTf, count, tfidf = stemVectorWeight(train_data, False, 'testpickle1.pickle', 'testpickle2.pickle', 'testpickle3.pickle')
    invIdx = generateInvertedIndex(count, XtrainTf, 'testpickle11.pickle')
    
    # Test on test data
    y_true = []
    y_pred = []
    
    for test_prompt, true_intent in test_data:
        result = searchIntent(invIdx, test_prompt, count, tfidf, train_data, skip_confirmation=True)
        
        if result:
            predicted_intent = train_data[result[0]][1]
        else:
            predicted_intent = "NONE"
        
        y_true.append(true_intent)
        y_pred.append(predicted_intent)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    print("RESULTS")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Per-class metrics
    print("PER-CLASS METRICS")
    
    unique_intents = sorted(set(y_true))
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_intents, zero_division=0
    )
    
    print(f"{'Intent':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    for i, intent in enumerate(unique_intents):
        print(f"{intent:<15} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} "
              f"{f1_per_class[i]:<12.4f} {support_per_class[i]:<10}")
    
    # Clean up test pickles after evaluation
    print("\nCleaning up test pickle files...")
    for pickle_file in glob.glob('testpickle*.pickle'):
        os.remove(pickle_file)
        print(f"Removed {pickle_file}")

if __name__ == "__main__":
    evaluate_intents()
