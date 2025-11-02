
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
1. Intent matching. [ ]
2. Identity matching. [ ]
3. Transactions. [ ]
4. Information retrieval & Question answering. [ ]
5. Small talk. [ ]
'''

'''
INITIAL IDEAS:
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
'''

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

if __name__ == "__main__":
    main()

# TODO: similarity-based intent matching

# TODO: small talk 

# TODO: discoverability

# TODO: user personalisation / memory

# TODO: Q&A with stocks & bonds dataset provided for checkpoint



