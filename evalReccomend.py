import os
import sys
import glob

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import readBookDescriptions, stemVectorWeight, generateInvertedIndex
from search import bookDescSearch

def evaluateRecommendations():
    print("Book Recommendation Evaluation")
    
    # Load book descriptions
    books = readBookDescriptions()
    print("Total books:", len(books))
    
    # Test queries matched to actual books in stock.json
    testQueries = [
        ("science fiction space exploration aliens civilizations", ["The 3 Body Problem", "Foundation", "Dune"]),
        ("fantasy magic wizards school hogwarts", ["Harry Potter and the Philosopher's Stone", "The Name of the Wind"]),
        ("mystery detective crime murder investigation", ["The Silent Patient", "Gone Girl", "The Girl with the Dragon Tattoo"]),
        ("dystopian totalitarian government surveillance oppression", ["1984", "The Handmaid's Tale", "Brave New World"]),
        ("world war two nazi holocaust france", ["The Book Thief", "All the Light We Cannot See", "The Nightingale"]),
        ("memoir autobiography true story childhood education", ["Educated", "Born a Crime", "The Glass Castle"]),
        ("epic fantasy adventure quest magic ring", ["The Lord of the Rings", "The Hobbit", "A Game of Thrones"]),
        ("cyberpunk hacker virtual reality technology", ["Neuromancer", "Snow Crash"]),
    ]
    
    # Clean up any existing pickle files
    print("Cleaning up existing pickle files...")
    for pickleFile in glob.glob('*.pickle'):
        os.remove(pickleFile)
    
    # Train on full dataset
    XtrainTf, count, tfidf = stemVectorWeight(books, False, 'testpickle7.pickle', 'testpickle8.pickle', 'testpickle9.pickle')
    invIdx = generateInvertedIndex(count, XtrainTf, 'testpickle13.pickle')
    
    # Evaluate recommendations
    correct = 0
    
    print("\nTesting Queries:")
    
    for query, relevantBooks in testQueries:
        # Get recommendations
        results = bookDescSearch(books, query, count, tfidf, invIdx)
        
        if results:
            # Get top recommendation
            topDocId = results[0][0]
            topBookTitle = books[topDocId][1]
            
            # Check if relevant
            if topBookTitle in relevantBooks:
                print("Valid")
                correct += 1
            else:
                print("Fail")
    
    print("\nResults:")
    accuracy = correct / len(testQueries)
    print("Top-1 Accuracy:", accuracy, f"({correct}/{len(testQueries)})")
    print("(Percentage of queries where top recommendation is relevant)")
    
    # Clean up test pickle files
    print("\nCleaning up test pickle files...")
    for pickleFile in glob.glob('testpickle*.pickle'):
        os.remove(pickleFile)

if __name__ == "__main__":
    evaluateRecommendations()
