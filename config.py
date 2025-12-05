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

confidenceThresholdQA = 0.6
confidenceThresholdQAconfirm = 0.4
confidenceThresholdIntent = 0.5
confidenceThresholdIntentconfirm = 0.3
