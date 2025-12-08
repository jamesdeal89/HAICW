# ====== EVALUATION OF QUESTION-ANSWERING ACCURACY ======
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import readQaCsv, stemVectorWeight, generateInvertedIndex
from search import question

def evaluateQA():
    print("Q&A System Evaluation")
    
    # Load dataset
    qa = readQaCsv()
    print("Dataset size:", len(qa))
    
    # Clean up any existing pickle files
    print("Cleaning up existing pickle files...")
    for pickleFile in glob.glob('*.pickle'):
        os.remove(pickleFile)
    
    # Train on full dataset
    XtrainTf, count, tfidf = stemVectorWeight(qa, True, 'testpickle4.pickle', 'testpickle5.pickle', 'testpickle6.pickle')
    invIdx = generateInvertedIndex(count, XtrainTf, 'testpickle12.pickle')
    
    # Evaluate on full dataset
    successful = 0
    errorIndicators = ["I don't have enough information", "Try rephrasing"]
    
    for entry in qa:
        testQuestion = entry[0]
        
        # Get predicted answer
        predictedAnswer = question(qa, testQuestion, count, tfidf, invIdx)
        
        # Check if valid answer returned
        isValid = not any(indicator in predictedAnswer for indicator in errorIndicators)
        
        if isValid:
            successful += 1
    
    # Calculate success rate
    successRate = successful / len(qa)
    
    print("\nResults:")
    print("Success Rate:", successRate, f"({successful}/{len(qa)})")
    print("(Success = system returned an answer, not an error message)")
    
    # Clean up test pickle files
    print("\nCleaning up test pickle files...")
    for pickleFile in glob.glob('testpickle*.pickle'):
        os.remove(pickleFile)

if __name__ == "__main__":
    evaluateQA()
