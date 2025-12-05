import re
import os
import json

from dataAccess import readName, saveName, resetSession
from utils import confirmation
from nlg import getReferringExpression, addDiscourseMarker, generateSuggestion
from context import updateContext, getContext

'''
Simple handler for when the user thanks the chatbot
'''
def thank():
    import random
    
    name = readName()
    
    if name:
        responses = [
            f"You're very welcome, {name}! Happy to help.",
            f"My pleasure, {name}! Let me know if you need anything else.",
            f"Anytime, {name}! Glad I could assist.",
            f"You're welcome, {name}! It's what I'm here for.",
            f"No problem at all, {name}! Feel free to ask me anything else.",
        ]
    else:
        responses = [
            "You're welcome! Happy to help.",
            "My pleasure! Let me know if you need anything else.",
            "Anytime! Glad I could assist.",
            "You're welcome! It's what I'm here for.",
            "No problem at all! Feel free to ask me anything else.",
        ]
    
    return random.choice(responses)

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
                r"(?i)\bwhen\s+(.+?)\s+(?:i|me)\b(?:\s+\w)?\s+feel\s+(.+)",
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
            if word.lower() in referenceMap.keys():
                reason += referenceMap[word.lower()] + " "
            else:
                reason += word + " "

        # [:-1] indexing on reason is simply to remove trailing whitespace before ?
        response += f"Why do you feel {emotionWithReason.group(1)} when {reason[:-1]}? "
    elif emotionWithReason1:
        reason = ""
        # instead of split(' '), this will also remove punctuation when tokenising
        for word in re.findall(r"\w+",emotionWithReason1.group(1)):
            if word.lower() in referenceMap.keys():
                reason += referenceMap[word.lower()] + " "
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
    "Let me walk you through my features:\n\n"
    "    1. Book orders - I can help you order a book for either home delivery or pick-up from one of our locations.\n\n",
    "   2. Book reccomendation - if you're not sure what you want to order, you can ask me about genres and authors and I'll explain their work.\n\n",
    "   3. Location information - I can list all our locations, describe the facilties of specific stores and let you know their opening times.\n\n",
    "   4. Book and literary information QA - you can ask me about writing, literature, and authors and I'll explain them.\n\n",
    "   5. Memory - I will remember your name and address you as such if you inform me of it!\n\n",
    "   6. Order tracking - you can ask me to check on your previous orders to see the book, quantities, when you need to pick them up, or how much you paid.\n\n"
    "    7. Small talk - I can handle basic small talk if you'd like to chat.\n\n",
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
            return f"Thanks for letting me know your name, {readName()}. I'll remember that. If you'd like to find out more about what I can do, ask 'what can you do?'"

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
Uses genre-based recommendations from the available stock.
'''
def reccomend(prompt='', bookDesc=None, countBookDesc=None, tfidfBookDesc=None, invIdxBookDesc=None):
    from dataAccess import getGenres, getGenreBooks, getBookByTitle
    from search import bookDescSearch
    import random
    
    updateContext('lastIntent', 'recommend')
    
    from context import resolveEllipsis
    
    # Ask user for recommendation type
    print("Would you like genre-based or recommendation-based suggestions?")
    print("[1] Genre-based (browse by genre)")
    print("[2] Recommendation-based (describe what you're looking for)")
    
    choiceInput = input("Please enter your choice (1 or 2, or QUIT to exit): ")
    if choiceInput.lower() == "quit":
        exit()
    
    if choiceInput == "2" and bookDesc and countBookDesc and tfidfBookDesc and invIdxBookDesc:
        # Recommendation-based search using description matching
        # Loop to allow rephrasing
        while True:
            print("Please describe the type of book you're looking for:")
            descInput = input("Please enter your prompt (QUIT to exit): ")
            if descInput.lower() == "quit":
                exit()
            userDesc = resolveEllipsis(descInput)
            
            results = bookDescSearch(bookDesc, userDesc, countBookDesc, tfidfBookDesc, invIdxBookDesc)
            
            if results:
                # Try first recommendation
                docId, score = results[0]
                bookTitle = bookDesc[docId][0]
                bookData = getBookByTitle(bookTitle)
                
                if bookData:
                    updateContext('lastBook', bookTitle)
                    print(f"\nI recommend '{bookTitle}' by {bookData['author']}!")
                    ref = getReferringExpression(bookTitle, 'book', False)
                    print(f"{ref.capitalize()} is a {bookData['genre']} book with {bookData['pages']} pages, priced at £{bookData['price']:.2f}.")
                    print(f"We have {bookData['count']} copies in stock.")
                    print(f"\nDescription: {bookData['description']}")
                    
                    # Ask for confirmation using utility
                    print("\nDoes this sound like what you're looking for?")
                    from utils import confirmation
                    if confirmation():
                        # User accepted - exit the loop
                        print("Great! Let me know if you'd like to order this book or if you need anything else.")
                        break
                    else:
                        # User declined - try second recommendation if available
                        if len(results) > 1:
                            docId2, score2 = results[1]
                            bookTitle2 = bookDesc[docId2][0]
                            bookData2 = getBookByTitle(bookTitle2)
                            
                            if bookData2:
                                updateContext('lastBook', bookTitle2)
                                print(f"\nAlright, how about '{bookTitle2}' by {bookData2['author']}?")
                                ref2 = getReferringExpression(bookTitle2, 'book', False)
                                print(f"{ref2.capitalize()} is a {bookData2['genre']} book with {bookData2['pages']} pages, priced at £{bookData2['price']:.2f}.")
                                print(f"We have {bookData2['count']} copies in stock.")
                                print(f"\nDescription: {bookData2['description']}")
                                
                                # Ask for confirmation again
                                print("\nDoes this sound better?")
                                if confirmation():
                                    # User accepted second recommendation - exit loop
                                    print("Great! Let me know if you'd like to order this book or if you need anything else.")
                                    break
                                else:
                                    # User declined both - loop back to ask for new description
                                    print("I apologize, but I'm having trouble finding what you're looking for. Could you try describing the book differently?")
                                    continue
                            else:
                                print("I apologize, but I'm having trouble finding what you're looking for. Could you try describing the book differently?")
                                continue
                        else:
                            print("I apologize, but I'm having trouble finding what you're looking for. Could you try describing the book differently?")
                            continue
                else:
                    from nlg import generateContextualError
                    print(generateContextualError('book_not_found', bookTitle))
                    continue
            else:
                from nlg import generateContextualError
                print(generateContextualError('generic'))
                continue
    else:
        # Genre-based search (original implementation)
        genres = getGenres()
        matchedGenre = None
        
        if prompt:
            resolvedInput = resolveEllipsis(prompt)
            for genre in genres:
                if genre.lower() in resolvedInput.lower() or resolvedInput.lower() in genre.lower():
                    matchedGenre = genre
                    break
        
        if not matchedGenre:
            print("What genre are you interested in?")
            print(generateSuggestion('available_genres', genres))
            
            genreInput = input("Please enter your prompt (QUIT to exit): ")
            if genreInput.lower() == "quit":
                exit()
            
            resolvedInput = resolveEllipsis(genreInput)
            
            for genre in genres:
                if genre.lower() in resolvedInput.lower() or resolvedInput.lower() in genre.lower():
                    matchedGenre = genre
                    break
        
        if matchedGenre:
            updateContext('lastGenre', matchedGenre)
            books = getGenreBooks(matchedGenre)
            if books:
                recommended = random.choice(books)
                title = recommended['name']
                updateContext('lastBook', title)
                print(f"\nI recommend '{title}' by {recommended['author']}!")
                ref = getReferringExpression(title, 'book', False)
                print(f"{ref.capitalize()} is a {matchedGenre} book with {recommended['pages']} pages, priced at £{recommended['price']:.2f}.")
                print(f"We have {recommended['count']} copies in stock.")
            else:
                from nlg import generateContextualError
                print(generateContextualError('book_not_found', f"any {matchedGenre} books"))
        else:
            from nlg import generateContextualError
            error = generateContextualError('generic')
            print(f"{error} Please try again with one of the available genres.")

'''
Handles when a user's intent is to check their existing orders.
'''
def check():
    from dataAccess import getOrdersJSON, readName
    from datetime import datetime
    
    updateContext('lastIntent', 'check')
    
    orders = getOrdersJSON()
    
    if not orders or not orders.get('orders'):
        print("There are no orders in the system yet.")
        return
    
    userName = readName()
    
    if userName:
        filteredOrders = [order for order in orders['orders'] if order.get('name', '').lower() == userName.lower()]
        
        if not filteredOrders:
            print(f"No orders found for {userName}.")
            return
        
        print(f"Orders for {userName}:")
        for i, order in enumerate(filteredOrders, 1):
            print(f"\nOrder {i}:")
            print(f"  Book: {order['title']}")
            print(f"  Quantity: {order['quantity']}")
            print(f"  Cost: £{order['cost']:.2f}")
            
            if order.get('pickup'):
                print(f"  Pickup Location: {order['address']}")
                if order.get('date'):
                    orderDate = datetime.fromtimestamp(order['date'])
                    print(f"  Pickup Date: {orderDate.strftime('%d/%m/%Y')}")
                if order.get('time') is not None:
                    hour = order['time']
                    print(f"  Pickup Time: {hour:02d}:00")
            else:
                print(f"  Delivery Address: {order['address']}")
    else:
        print("All orders in the system:")
        for i, order in enumerate(orders['orders'], 1):
            print(f"\nOrder {i}:")
            print(f"  Customer: {order.get('name', 'Unknown')}")
            print(f"  Book: {order['title']}")
            print(f"  Quantity: {order['quantity']}")
            print(f"  Cost: £{order['cost']:.2f}")
            
            if order.get('pickup'):
                print(f"  Pickup Location: {order['address']}")
                if order.get('date'):
                    orderDate = datetime.fromtimestamp(order['date'])
                    print(f"  Pickup Date: {orderDate.strftime('%d/%m/%Y')}")
                if order.get('time') is not None:
                    hour = order['time']
                    print(f"  Pickup Time: {hour:02d}:00")
            else:
                print(f"  Delivery Address: {order['address']}")

'''
Handles when a user's intent is to query about available locations.
'''
def locations():
    from dataAccess import getAllLocations
    
    print("Our bookstores can be found in the following locations:")
    for location in getAllLocations():
        print(f"Location: {location[0]}, Address: {location[1]}")

'''
Handles when a user's intent is to query about openining times / dates.
'''
def opening(prompt=None):
    from dataAccess import getLocationsJSON, extractLocation
    from context import resolveEllipsis, updateContext
    
    updateContext('lastIntent', 'opening')
    
    locations = getLocationsJSON()
    
    location = None
    if prompt:
        location = extractLocation(prompt)
    
    if not location:
        print("Which location would you like to know the opening hours for?")
        locationInput = input("Please enter your prompt (QUIT to exit): ")
        if locationInput.lower() == "quit":
            exit()
        
        resolvedInput = resolveEllipsis(locationInput)
        location = extractLocation(resolvedInput)
    
    if location:
        for loc in locations['locations']:
            if loc['name'].lower() == location.lower():
                updateContext('lastLocation', loc['name'])
                days = loc['days']
                dayNames = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                openDays = [dayNames[i] for i, d in enumerate(days) if d == '1']
                
                print(f"\n{loc['name']} location:")
                print(f"Opening hours: {loc['open']}:00 - {loc['close']}:00")
                if openDays:
                    print(f"Open on: {', '.join(openDays)}")
                return
    
    print("I couldn't find that location. Here are our locations:")
    for loc in locations['locations']:
        print(f"  - {loc['name']}: {loc['address']}")

'''
Handles when a user's intent is to query about the bookstore's address.
'''
def address(prompt=None):
    from dataAccess import getLocationsJSON, extractLocation
    from context import resolveEllipsis, updateContext
    
    updateContext('lastIntent', 'address')
    
    locations = getLocationsJSON()
    
    location = None
    if prompt:
        location = extractLocation(prompt)
    
    if not location:
        print("Which location's address would you like?")
        locationInput = input("Please enter your prompt (QUIT to exit): ")
        if locationInput.lower() == "quit":
            exit()
        
        resolvedInput = resolveEllipsis(locationInput)
        location = extractLocation(resolvedInput)
    
    if location:
        for loc in locations['locations']:
            if loc['name'].lower() == location.lower():
                updateContext('lastLocation', loc['name'])
                print(f"\n{loc['name']} location:")
                print(f"Address: {loc['address']}")
                if 'email' in loc:
                    print(f"Email: {loc['email']}")
                return
    
    print("I couldn't find that location. Here are all our locations:")
    for loc in locations['locations']:
        print(f"\n{loc['name']}:")
        print(f"  Address: {loc['address']}")
        if 'email' in loc:
            print(f"  Email: {loc['email']}")

'''
Handles when a user's intent is to query about a location's facilities.
'''
def facilities(prompt=None):
    from dataAccess import getLocationsJSON, extractLocation
    from context import resolveEllipsis, updateContext
    
    updateContext('lastIntent', 'facilities')
    
    locations = getLocationsJSON()
    
    location = None
    if prompt:
        location = extractLocation(prompt)
    
    if not location:
        print("Which location's facilities would you like to know about?")
        locationInput = input("Please enter your prompt (QUIT to exit): ")
        if locationInput.lower() == "quit":
            exit()
        
        resolvedInput = resolveEllipsis(locationInput)
        location = extractLocation(resolvedInput)
    
    if location:
        for loc in locations['locations']:
            if loc['name'].lower() == location.lower():
                updateContext('lastLocation', loc['name'])
                print(f"\n{loc['name']} location facilities:")
                if 'cafe' in loc and loc['cafe']:
                    print("- Cafe available")
                if 'floors' in loc:
                    print(f"- {loc['floors']} floors")
                return
    
    print("I couldn't find that location. Here are our locations:")
    for loc in locations['locations']:
        print(f"\n{loc['name']}:")
        if 'cafe' in loc and loc['cafe']:
            print("- Cafe available")
        if 'floors' in loc:
            print(f"- {loc['floors']} floors")

def stockCheck(prompt=None):
    from dataAccess import fuzzySearchTitle, getLocationsAvailable, extractLocation
    import re
    
    updateContext('lastIntent', 'stockCheck')
    
    book = None
    location = None
    
    if prompt:
        locationExtract = r"(?i)\b(?:at|in|from)\s+([A-Za-z\s]+)(?:\?|$)"
        locationMatch = re.search(locationExtract, prompt)
        
        titleExtract = r"(?i)(?:can you tell me if|can i check if|can you check if|can i see if you have|is|check stock for|check if you have|check|do you have|where can i find|where can i get|which locations have|which stores have|where is|where do you have|what locations stock|availability of|available for|stock)\s+['\"]?([A-Za-z0-9'':,&() ]{3,})['\"]?(?:\s+(?:is in stock|in stock|available|stocked|is available))?(?:\s+(?:at|in|from)\s+[A-Za-z\s]+)?(?:\?|$)"
        titleMatch = re.search(titleExtract, prompt)
        
        if titleMatch:
            potentialTitle = titleMatch.group(1).strip()
            
            if locationMatch:
                potentialTitle = re.sub(r"(?i)\s+(?:at|in|from)\s+[A-Za-z\s]+(?:\?|$)", "", potentialTitle).strip()
                location = extractLocation(locationMatch.group(1))
            
            potentialTitle = re.sub(r"(?i)\s+(?:is in stock|in stock|available|stocked|is available)$", "", potentialTitle).strip()
            
            matches = fuzzySearchTitle(potentialTitle)
            if matches:
                if matches[0][1] <= 2:
                    book = matches[0][0]
                    updateContext('lastBook', book)
                elif matches[0][1] <= len(potentialTitle) // 3:
                    print(f"Did you mean '{matches[0][0]}'?")
                    if confirmation():
                        book = matches[0][0]
                        updateContext('lastBook', book)
    
    if not book:
        lastBook = getContext('lastBook')
        if lastBook:
            print(f"Check stock for {lastBook}?")
            if confirmation():
                book = lastBook
        
        if not book:
            print("Which book would you like to check stock for?")
            answer = input("Please enter your prompt (QUIT to exit): ")
            if answer.lower() == "quit":
                exit()
            
            matches = fuzzySearchTitle(answer)
            if matches:
                if matches[0][1] <= 2:
                    book = matches[0][0]
                    updateContext('lastBook', book)
                else:
                    print(f"Did you mean '{matches[0][0]}'?")
                    if confirmation():
                        book = matches[0][0]
                        updateContext('lastBook', book)
            
            if not book:
                print(f"Sorry, I couldn't find that book in our stock.")
                return
    
    availableLocations = getLocationsAvailable(book)
    
    if not availableLocations:
        print(f"'{book}' is currently not in stock at any location.")
        return
    
    if location:
        locationLower = location.lower()
        availableLocationsLower = [loc.lower() for loc in availableLocations]
        
        if locationLower in availableLocationsLower:
            properLocation = availableLocations[availableLocationsLower.index(locationLower)]
            print(f"Yes, '{book}' is in stock at {properLocation}.")
            updateContext('lastLocation', properLocation)
        else:
            print(f"No, '{book}' is not in stock at {location}.")
            print(f"However, it is available at: {', '.join(availableLocations)}")
    else:
        print(f"'{book}' is in stock at the following locations:")
        for loc in availableLocations:
            print(f"  - {loc}")

