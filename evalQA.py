import os
import sys
import glob

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import readQaCsv, stemVectorWeight, generateInvertedIndex
from search import question
from sklearn.model_selection import train_test_split

def evaluate_qa():
    print("Q&A SYSTEM EVALUATION")
    
    # Load data
    qa = readQaCsv()
    
    # Split into train/test (80/20)
    train_data, test_data = train_test_split(qa, test_size=0.2, random_state=42)
    
    print(f"\nDataset size: {len(qa)}")
    print(f"Training set: {len(train_data)}")
    print(f"Test set: {len(test_data)}")
    
    # Clean up any existing pickle files
    print("\nCleaning up existing pickle files...")
    for pickle_file in glob.glob('*.pickle'):
        os.remove(pickle_file)
    
    # Train on training data
    XtrainTf, count, tfidf = stemVectorWeight(train_data, True, 'testpickle4.pickle', 'testpickle5.pickle', 'testpickle6.pickle')
    invIdx = generateInvertedIndex(count, XtrainTf, 'testpickle12.pickle')
    
    # Evaluate - count successful retrievals (not error messages)
    successful = 0
    error_indicators = ["I don't have enough information", "Try rephrasing"]
    
    for entry in test_data:
        test_question = entry[0]
        
        # Get predicted answer using the question function
        predicted_answer = question(train_data, test_question, count, tfidf, invIdx)
        
        # Check if it returned a valid answer (not an error message)
        is_valid = not any(indicator in predicted_answer for indicator in error_indicators)
        
        if is_valid:
            successful += 1
    
    # Calculate metrics
    success_rate = successful / len(test_data)
    
    print("RESULTS")
    print(f"Success Rate: {success_rate:.4f} ({successful}/{len(test_data)})")
    print(f"(Success = system returned an answer, not an error message)")
    
    # Clean up test pickle files
    print("\nCleaning up test pickle files...")
    for pickle_file in glob.glob('testpickle*.pickle'):
        os.remove(pickle_file)
    
    print("\n")

if __name__ == "__main__":
    evaluate_qa()
