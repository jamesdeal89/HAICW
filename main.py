# ====== COMP3074 Coursework Chatbot ======
# Author: James Deal
# ID: 20551937

'''
NOTE: Allowed libraries include: 
- Python standard library.
- numpy,
- scipy,
- scikit-learn (and it's dependencies),
- nltk (EXCEPT nltk.chat)
'''

'''
CHECKLIST OF EXPECTED FEATURES:
1. Intent matching. [x]
2. Identity management. [x]
3. Transactions. [x]
4. Information retrieval & Question answering. [x]
5. Small talk. [x]
'''

'''
IDEA:
Book ordering system.
Transaction: ordering books, scheduling pickup/delivery.
Q&A: book recomendations, location opening times.
'''

'''
FOR THE CHECKPOINT:
Requirements:
Implement *similarity-based* intent matching, 
which routes the flow of the chatbot to functions for handling:
- Small talk,
- Discoverability (what can the bot do?),
- Identity management,
- Question answering.

They will test with:
Hi or Hello [greet the user and ask their name]
What is my name [tell the user their name]
How are you [answer]
What can you do [answer]
What are stocks and bonds [answer from QA dataset]

NOTE: For purposes of checkpoint, need to use *stocks QA dataset* provided, 
for final submission can change.

CHECKPOINT WAS COMPLETED AND BONUS MARKS RECEIVED.
'''

'''
NOTE on Q&A dataset:
Static dataset was partially genAI generated to bulk out examples. 
NONE of the code, processing, etc are AI generated.
NOTE on intents dataset:
Based on boostrapped manual examples,
Bulked out using confirmation:
    - if the user confirms a low confidence intent match, 
        - add their prompt and the confirmed intent to dataset.
'''

# Import from refactored modules
from preprocessing import readIntentsCsv, stemVectorWeight, generateInvertedIndex, readQaCsv
from search import searchIntent, question
from handlers import discover, small, identity, thank, reccomend
from orders import order

def main():
    print("======== STARTING BOOKSTORE CHATBOT ========")
    # NOTE: below ASCII art was generated via https://patorjk.com/software/taag/
    print("""
    ██████  ██       █████   ██████ ██   ██ ███████ ███    ███ ██ ████████ ██   ██ ███████     
    ██   ██ ██      ██   ██ ██      ██  ██  ██      ████  ████ ██    ██    ██   ██ ██          
    ██████  ██      ███████ ██      █████   ███████ ██ ████ ██ ██    ██    ███████ ███████     
    ██   ██ ██      ██   ██ ██      ██  ██       ██ ██  ██  ██ ██    ██    ██   ██      ██     
    ██████  ███████ ██   ██  ██████ ██   ██ ███████ ██      ██ ██    ██    ██   ██ ███████     
                                                                                            
                                                                                            
    ██████   ██████   ██████  ██   ██     ███████ ████████  ██████  ██████  ███████            
    ██   ██ ██    ██ ██    ██ ██  ██      ██         ██    ██    ██ ██   ██ ██                 
    ██████  ██    ██ ██    ██ █████       ███████    ██    ██    ██ ██████  █████              
    ██   ██ ██    ██ ██    ██ ██  ██           ██    ██    ██    ██ ██   ██ ██                 
    ██████   ██████   ██████  ██   ██     ███████    ██     ██████  ██   ██ ███████            
                                                                                            
                                                                                            
     ██████ ██   ██  █████  ████████ ██████   ██████  ████████                                 
    ██      ██   ██ ██   ██    ██    ██   ██ ██    ██    ██                                    
    ██      ███████ ███████    ██    ██████  ██    ██    ██                                    
    ██      ██   ██ ██   ██    ██    ██   ██ ██    ██    ██                                    
     ██████ ██   ██ ██   ██    ██    ██████   ██████     ██                                    
    """)
    print("Welcome to BlackSmith's Bookstore Chatbot!")

    # intent matching initialisations
    intents = readIntentsCsv()
    XtrainTf, count, tfidf = stemVectorWeight(intents, False, "XtrainTf.pickle", "count.pickle", "tfidf.pickle")
    invIdxIntents = generateInvertedIndex(count, XtrainTf, "invIdx.pickle")

    # QA search initialisations
    qa = readQaCsv()
    XtrainTfQa, countQa, tfidfQa = stemVectorWeight(qa, True, "XtrainTfQa.pickle", "countQa.pickle", "tfidfQa.pickle")
    invIdxQa = generateInvertedIndex(countQa, XtrainTfQa, "invIdxQa.pickle")

    while True:
        prompt = input("Please enter your prompt (QUIT to exit): ")
        if prompt.lower() == "quit":
            break
        intentResult = searchIntent(invIdxIntents, prompt, count, tfidf, XtrainTf, intents)
        if intentResult:
            intent = intents[intentResult[0]][1]
            if intent == "discover":
                discover()
            elif intent == "small":
                print(small(prompt))
            elif intent == "question":
                print(question(qa, prompt, countQa, tfidfQa, XtrainTfQa, invIdxQa))
            elif intent == "identity":
                print(identity(prompt))
            elif intent == "thank":
                print(thank())
            elif intent == "reccomend":
                reccomend()
            elif intent == "order":
                order(prompt, intents, count, tfidf, XtrainTf, invIdxIntents, 
                      qa, countQa, tfidfQa, XtrainTfQa, invIdxQa)
            else:
                print("I'm not sure I understand. Could you try re-wording that?")
        else:
            print("I'm not sure I understand. Could you try re-wording that?")

        
    print("Goodbye!")

if __name__ == "__main__":
    main()
