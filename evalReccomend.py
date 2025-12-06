import os
import sys
import glob

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import readBookDescriptions, stemVectorWeight, generateInvertedIndex
from search import bookDescSearch

def evaluate_recommendations():
    print("BOOK RECOMMENDATION EVALUATION")
    
    # Load book descriptions
    books = readBookDescriptions()
    
    print(f"\nTotal books: {len(books)}")
    
    # Test queries matched to actual books in stock.json
    test_queries = [
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
    print("\nCleaning up existing pickle files...")
    for pickle_file in glob.glob('*.pickle'):
        os.remove(pickle_file)
    
    # Train on full dataset
    XtrainTf, count, tfidf = stemVectorWeight(books, False, 'testpickle7.pickle', 'testpickle8.pickle', 'testpickle9.pickle')
    invIdx = generateInvertedIndex(count, XtrainTf, 'testpickle13.pickle')
    
    # Evaluate - check if top recommendation is relevant
    correct = 0
    
    print("RESULTS")
    
    for query, relevant_books in test_queries:
        # Get recommendations
        results = bookDescSearch(books, query, count, tfidf, invIdx)
        
        if results:
            # Get top recommendation
            top_doc_id = results[0][0]
            top_book_title = books[top_doc_id][1]
            
            # Check if it's relevant
            if top_book_title in relevant_books:
                print("Valid")
                correct += 1
            else:
                print("Fail")
    
    accuracy = correct / len(test_queries)
    print(f"Top-1 Accuracy: {accuracy:.4f} ({correct}/{len(test_queries)})")
    print(f"(Percentage of queries where top recommendation is relevant)")
    
    # Clean up test pickle files
    print("\nCleaning up test pickle files...")
    for pickle_file in glob.glob('testpickle*.pickle'):
        os.remove(pickle_file)
    
    print("\n")

if __name__ == "__main__":
    evaluate_recommendations()
