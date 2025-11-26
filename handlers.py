import re
import os
import json

from data_access import readName, saveName, resetSession
from utils import confirmation

'''
Simple handler for when the user thanks the chatbot
'''
def thank():
    if readName():
        print(f"You are very welcome, {readName()}! Glad to help.")
    else:
        print(f"You're welcome! Glad to help.")

'''
Small talk approach is based on 'ELIZA' from the course's recomended reading:
Pattern match prompt to identify keywords which can be saved and referred to in NLG template responses.
If no match, resort to generic response like "Why do you say that?" or "Tell me more".
Allow multiple matches from a single user prompt - gradually builds up a large response if the user's prompt was multi-faceted.
'''
def small(prompt):

    # List of regex patterns to match and capture specific keywords.
    # These keywords, depending on type, will be inserted into template responses. 

    # \b word boundary, ensures matches whole words
    # \s whitespace chars
    # \w word
    # (?:...) create non-capturing group

    patterns = [r"(?i)\b(?:feel|feeling)\s+(\w+)",
                r"(?i)\b(?:I|me)\b(?:\s+\w)?\s+feel\s+(.+?)\s+when\s+(.+)",
                r"(?i)\bwhen\s+(.+?)\s+(it\smakes\s)(?:i|me)\b(?:\s+\w)?\s+feel\s+(.+)",
                r"(?i)\b(?:how\sare\syou|how'?s\sit\sgoing|how'?s\sthings|what'?s\snew|how\shave\syou\sbeen)\b",
                r"(?i)\b(?:hi|hey|hello|howdy|greetings|good\s+(?:morning|afternoon|evening|day)|what'?s\sup|sup)\b"]

    emotion = re.search(patterns[0], prompt)
    emotionWithReason = re.search(patterns[1], prompt)
    emotionWithReason1 = re.search(patterns[2], prompt)
    howAre = re.search(patterns[3], prompt)
    greet = re.search(patterns[4], prompt)

    # Map for entity reference words, so they can be 'flipped' when the system uses them.
    # e.g: 'I feel sad when people laugh at me' -> 'why do you feel sad when people laugh at you'
    referenceMap = {
        'me': 'you',
        'i': 'you',
        'my': 'your',
    }

    # build up response gradually
    response = ""

    if greet:
        if readName():
            response += f"Hi, {readName()}. "
        else:
            print("Hello! ", end='')
            # re-use the identity function by putting a pseudo-prompt which will make the system ask for the user's name
            print(identity("what is my name"), end='')

    if emotionWithReason:
        reason = ""
        for word in re.findall(r"\w+",emotionWithReason.group(2)):
            if word in referenceMap.keys():
                reason += referenceMap[word] + " "
            else:
                reason += word + " "

        # [:-1] indexing on reason is simply to remove trailing whitespace before ?
        response += f"Why do you feel {emotionWithReason.group(1)} when {reason[:-1]}? "
    elif emotionWithReason1:
        reason = ""
        # instead of split(' '), this will also remove punctuation when tokenising
        for word in re.findall(r"\w+",emotionWithReason1.group(1)):
            if word in referenceMap.keys():
                reason += referenceMap[word] + " "
            else:
                reason += word + " "

        response += f"Why do you feel {emotionWithReason1.group(2)} when {reason[:-1]}? "
    
    elif emotion:
        response += f"Tell me more about why you feel {emotion.group(1)}. "

    if howAre:
        if readName():
            response += f"I'm doing well! How about you, {readName()}? "
        else:
            response += "I'm doing well! How about you? "
    
    if not response and not greet:
        response += "Tell me more."

    return response

'''
Handler for when intent is detected to be 'discover' which means the user is asking about what the chatbot can do.
Prints out a detailed explanation of capabilties to make the chatbot less of a 'black box'.
'''
def discover():
    print(
    "I can help with many things!\n",
    "Let me walk you through my features:\n"
    "   1. Book orders - I can help you order a book and set a delivery or pick-up time.\n",
    "   2. Book information - if you're not sure what you want to order, you can ask me about genres and authors and I'll recommend you a book.\n",
    "   3. Memory - I will remember your name and address you as such if you inform me of it!\n",
    "   4. Small talk - I can handle basic small talk if you'd like to chat.\n",
    "I hope this helps you converse with me effectively!")

def identity(prompt):
    # Use regular expression to determine specific intent (either they want to set the name or recall it)
    reIntents = [r"(?i)\b(?:my name is|call me|i am|i'm)\s+([A-Za-z][A-Za-z' -]*)",
                 r"(?i)\b(?:what\s+is\s+my\s+name|do\s+you\s+know\s+my\s+name|who\s+am\s+i)\b",
                 r"(?i)\b(?:forget\s+me|forget\s+my\s+(?:info|information|name|data)|delete\s+my\s+(?:info|information|name|data)|remove\s+my\s+data|erase\s+my\s+information)\b"]
    name = re.search(reIntents[0], prompt)
    recall = re.search(reIntents[1], prompt)
    forget = re.search(reIntents[2], prompt)
    if name:
        saveName(name.group(1))
        return f"Thanks for letting me know your name, {name.group(1)}. I'll remember that."
    elif recall:
        name = readName()
        if name:
            return f"I remember your name is {name}, of course!"
        else:
            # Need to a separate input here as they may respond with simply 'Joe' which the intents input loop may not be able to handle.
            print("I'm sorry, I don't think you've told me your name before. What is your name?")
            answer = input("Please enter your prompt (QUIT to exit): ")
            if answer.lower() == "quit":
                exit()
            uName = re.search(reIntents[0],answer)
            if not uName and len(answer.strip().split(' ')) == 1:
                saveName(answer.strip())
            else:
                saveName(uName.group(1))
            return f"Thanks for letting me know your name, {readName()}. I'll remember that."

    elif forget:
        # Confirm user's intention to remove their data from memory
        print("You want me to forget your information I've saved up until now?")
        if confirmation():
            resetSession()
            return "Okay I've forgotten any information I remembered about you."
        else:
            return "Okay, I won't forget your information."

        

    else:
        return "I'm sorry, I don't understand your indentity related request. Perhaps try re-wording your prompt."

'''
Handles when the user wants a reccomendation for a book.
'''
def reccomend():
    # Get list of genres actually in the stock list
    from data_access import getGenres
    genres = getGenres()

'''
Handles when a user's intent is to check their existing orders.
'''
def check():
    pass

'''
Handles when a user's intent is to query about openining times / dates.
'''
def opening():
    pass

'''
Handles when a user's intent is to query about the bookstore's address.
'''
def address():
    pass

'''
Handles when a user's intent is to query about a location's facilities.
'''
def facilities():
    pass
