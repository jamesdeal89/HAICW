# ------ Orders/Transactions ------

import re
import datetime

from dataAccess import (
    fuzzySearchTitle, stockCheck, getPrice, getAllLocations, extractLocation,
    isLocOpenOnDate, isLocOpenAtTime, storeOrder, getISBNbyTitle, readName, storeFeedback, getLocationsAvailable
)
from utils import confirmation, wordToInt, getUnixEpochTimestamp
from handlers import identity, small, discover, thank, reccomend
from search import searchIntent, question
from nlg import aggregateOrderDetails, generateContextualError, addDiscourseMarker
from context import updateContext, resolveEllipsis

# Global variables for intent matching during transactions
# These will be set by order() function from main's context
_intents = None
_count = None
_tfidf = None
_invIdxIntents = None
_qa = None
_countQa = None
_tfidfQa = None
_invIdxQa = None
_bookDesc = None
_countBookDesc = None
_tfidfBookDesc = None
_invIdxBookDesc = None

def handleInputWithIntents(userInput: str, expectedType: str = None):
    '''
    Intercepts user input during transactions.
    First tries to process as transaction data (based on expectedType).
    Only if input doesn't make sense for the transaction, check for non-order intents.
    
    expectedType can be: 'book', 'quantity', 'location', 'date', 'time', 'pickup_delivery', 'general'
    
    Returns: (processed_input, should_retry)
    - If transaction-relevant: (userInput, False) 
    - If intent handled: (None, True) - signals to re-ask
    - If quit/cancel: (userInput, False) - let caller handle
    '''
    if not userInput:
        return userInput, False
    
    # Always allow quit and cancel
    if userInput.lower() in ['quit', 'cancel'] or 'cancel' in userInput.lower():
        return userInput, False
    
    # Resolve ellipsis BEFORE checking transaction relevance
    # This ensures "it", "that one", etc. are resolved to actual book names
    userInput = resolveEllipsis(userInput)
    
    # Check if input seems transaction-relevant based on context
    isTransactionRelevant = False
    
    if expectedType == 'quantity':
        # Check if it contains numbers or number words
        if re.search(r'\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve', userInput, re.IGNORECASE):
            isTransactionRelevant = True
    elif expectedType == 'book':
        # If it's a reasonable length for a book title (more than 2 chars)
        # But exclude common intent patterns that shouldn't be treated as book titles
        intentPatterns = r'(?i)^(recommend|suggest|what|where|when|why|how|tell|show|list|can you|do you|are you|help|discover)'
        if len(userInput.strip()) > 2 and not re.match(r'^(yes|no|yeah|nope|yep)$', userInput, re.IGNORECASE) and not re.search(intentPatterns, userInput):
            isTransactionRelevant = True
    elif expectedType == 'location':
        # Check if 'list' command or if it might be a location name
        if 'list' in userInput.lower() or len(userInput.split()) <= 3:
            isTransactionRelevant = True
    elif expectedType == 'pickup_delivery':
        # Check for pickup/delivery keywords
        if re.search(r'pickup|pick-up|delivery|deliver|home|store', userInput, re.IGNORECASE):
            isTransactionRelevant = True
    elif expectedType == 'date':
        # Check for date-related content
        if re.search(r'\d+|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december', userInput, re.IGNORECASE):
            isTransactionRelevant = True
    elif expectedType == 'time':
        # Check for time-related content
        if re.search(r'\d+|am|pm|noon|midnight|morning|afternoon|evening|lunchtime', userInput, re.IGNORECASE):
            isTransactionRelevant = True
    elif expectedType == 'general':
        # For yes/no confirmations, always process as transaction
        if re.match(r'^(yes|no|yeah|nope|yep|y|n)$', userInput, re.IGNORECASE):
            isTransactionRelevant = True
    
    # If input seems transaction-relevant, return it for normal processing
    if isTransactionRelevant:
        return userInput, False
    
    # Input doesn't seem transaction-relevant, check if it matches an intent
    if not _intents:
        return userInput, False
        
    intentResult = searchIntent(_invIdxIntents, userInput, _count, _tfidf, _intents)
    
    if intentResult:
        intent = _intents[intentResult[0]][1]
        
        # Handle intents that can be processed mid-transaction
        if intent == "small":
            print(small(userInput))
            print("\nNow, back to your order...")
            return None, True  # Signal to re-ask the question
        elif intent == "discover":
            discover()
            print("\nNow, back to your order...")
            return None, True
        elif intent == "identity":
            print(identity(userInput))
            print("\nNow, back to your order...")
            return None, True
        elif intent == "thank":
            print(thank())
            print("\nNow, back to your order...")
            return None, True
        elif intent == "question":
            # Handle Q&A during transaction
            if _qa:
                print(question(_qa, userInput, _countQa, _tfidfQa, _invIdxQa))
                print("\nNow, back to your order...")
                return None, True
        elif intent == "reccomend":
            # Handle book recommendations during transaction
            if _bookDesc:
                reccomend(userInput, _bookDesc, _countBookDesc, _tfidfBookDesc, _invIdxBookDesc)
                print("\nNow, back to your order...")
                return None, True
        # For "order" intent, don't interrupt - treat as normal input (avoid nested orders)
    
    # Didn't match any special intent, return for normal processing
    return userInput, False

def detectCorrection(userInput: str) -> tuple[bool, str, str]:
    """
    Detects if user is trying to correct a previous input.
    Returns tuple: (isCorrection: bool, correctionType: str, newValue: any)
    """
    inputLower = userInput.lower().strip()
    
    # Correction patterns
    quantityCorrection = re.search(r'(?:sorry|no|wait|actually|i meant)\s+(?:i meant |i wanted |it\'?s |make that )?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:copies|copy|books?)?', inputLower)
    if quantityCorrection:
        numStr = quantityCorrection.group(1)
        try:
            newQty = int(numStr) if numStr.isdigit() else wordToInt(numStr)
            return (True, 'quantity', newQty)
        except:
            pass
    
    bookCorrection = re.search(r'(?:sorry|no|wait|actually|i meant)\s+(?:i meant |i wanted |it\'?s |make that )?["\']?([A-Za-z0-9\s:,\-\'&()]+?)["\']?\s*(?:instead|actually|please)?$', inputLower)
    if bookCorrection and not quantityCorrection:
        return (True, 'book', bookCorrection.group(1).strip())
    
    locationCorrection = re.search(r'(?:sorry|no|wait|actually|i meant)\s+(?:i meant |i wanted |it\'?s |make that |from )?([A-Za-z\s]+?)(?:\s+(?:instead|actually|please|location|store))?$', inputLower)
    if locationCorrection and any(word in inputLower for word in ['location', 'store', 'pickup', 'from']):
        return (True, 'location', locationCorrection.group(1).strip())
    
    return (False, None, None)

def collectFeedback(lastPrompt: str = "") -> None:
    print("\nWe'd appreciate your feedback to help us improve!")
    print("On a scale of 1-5, how would you rate your experience? (1=poor, 5=excellent)")
    while True:
        ratingInput = input("Please enter your prompt (QUIT to exit): ")
        if ratingInput.lower() == "quit":
            exit()
        try:
            rating = int(ratingInput)
            if 1 <= rating <= 5:
                break
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number between 1 and 5.")
    
    print("Would you like to provide additional comments about your experience? (yes/no)")
    if confirmation():
        print("Please share your feedback:")
        comments = input("Please enter your prompt (QUIT to exit): ")
        if comments.lower() == "quit":
            exit()
    else:
        comments = ""
    
    storeFeedback(rating, comments, lastPrompt)
    print("Thank you for your feedback!")

def getBookSelection(prompt: str) -> str | None:
    reTitleExtract = r"(?i)\b(?:order|buy|get|purchase|place)\b.*?\b([A-Za-z0-9'':,&() ]{3,}?)\b(?=(?:\s+(?:for|from|to|at|in|pickup|delivery|delivered|store)\b|[.?!,;]|$))"
    title = re.search(reTitleExtract, prompt)
    
    matches = []
    if title:
        matches = fuzzySearchTitle(title.group(1))
    if matches:
        if matches[0][1] <= 2:
            book = matches[0][0]
            updateContext('lastBook', book)
            print(f"Ordering: {book}")
            return book
        else:
            print(f"Okay! So you'd like to order: {matches[0][0]}?")
            if confirmation():
                book = matches[0][0]
                return book
            elif len(matches) > 1:
                print(f"Did you mean: {matches[1][0]}?")
                if confirmation():
                    book = matches[1][0]
                    return book
                elif len(matches) > 2:
                    print(f"Or perhaps: {matches[2][0]}?")
                    if confirmation():
                        book = matches[2][0]
                        return book
    
    print("You'd like to place an order for which book? (please enter just the title)")
    while True:
        answer, shouldRetry = handleInputWithIntents(input("Please enter your prompt (QUIT to exit) (CANCEL to cancel order): "), 'book')
        if shouldRetry:
            continue
        if answer.lower() == "quit":
            exit()
        if 'cancel' in answer.lower():
            return None
        break
    
    # Resolve ellipsis (e.g., "that one", "it") before fuzzy searching
    answer = resolveEllipsis(answer)
    
    # If ellipsis resolution added "order" prefix, extract the book title
    orderMatch = re.search(reTitleExtract, answer)
    if orderMatch:
        answer = orderMatch.group(1)
    
    attempts = 0
    while True:
        matches = fuzzySearchTitle(answer)
        if matches:
            if matches[0][1] <= 2:
                book = matches[0][0]
                print(f"Ordering: {book}")
                return book
            else:
                print(f"Okay! So you'd like to order: {matches[0][0]}?")
                if confirmation():
                    book = matches[0][0]
                    return book
                elif len(matches) > 1:
                    print(f"Did you mean: {matches[1][0]}?")
                    if confirmation():
                        book = matches[1][0]
                        return book
                    elif len(matches) > 2:
                        print(f"Or perhaps: {matches[2][0]}?")
                        if confirmation():
                            book = matches[2][0]
                            return book
                # All suggestions declined, cancel order
                print("Unable to find the book you're looking for.")
                print("Cancelling order.")
                return None
        else:
            # No fuzzy search match, so see if the user mistakenly tried to enter another kind of prompt.
            _, shouldRetry = handleInputWithIntents(answer, None)
            if shouldRetry:
                # Intent was handled, don't increment attempts
                print("\nNow, which book would you like to order?")
                while True:
                    answer, shouldRetry = handleInputWithIntents(input("Please enter your prompt (QUIT to exit) (CANCEL to cancel order): "), 'book')
                    if shouldRetry:
                        continue
                    if answer.lower() == "quit":
                        exit()
                    if 'cancel' in answer.lower():
                        return None
                    break
            else:
                # No intent matched, increment attempts
                attempts += 1
                if attempts >= 3:
                    print("Sorry, we don't stock that book.")
                    print("Cancelling order.")
                    return None
                else:
                    print(generateContextualError('book_not_found', answer))
                    print(addDiscourseMarker('continuation', "Please try again."))
                    while True:
                        answer, shouldRetry = handleInputWithIntents(input("Please enter your prompt (QUIT to exit) (CANCEL to cancel order): "), 'book')
                        if shouldRetry:
                            continue
                        if answer.lower() == "quit":
                            exit()
                        if 'cancel' in answer.lower():
                            return None
                        break

def getQuantitySelection(prompt: str, book: str) -> int | None:
    reQuantityExtract = r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen| \
                        sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)\b|\b(\d+)\b"
    numbers = re.search(reQuantityExtract, prompt)
    
    skip = False
    if numbers:
        if numbers.group(1):
            quant = wordToInt(numbers.group(1).lower())
        else:
            quant = numbers.group(2)
        
        allNumbers = re.findall(reQuantityExtract, prompt)
        
        needsConfirmation = False
        
        if len(allNumbers) > 1:
            print("I found multiple numbers in your request. Which quantity did you mean?")
            for i, num in enumerate(allNumbers, 1):
                numValue = wordToInt(num[0].lower()) if num[0] else num[1]
                print(f"{i}. {numValue} copies")
            
            choice = input("Please enter the number (1, 2, etc.): ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(allNumbers):
                    selectedNum = allNumbers[idx]
                    quant = wordToInt(selectedNum[0].lower()) if selectedNum[0] else selectedNum[1]
                    return int(quant)
            except ValueError:
                print("Sorry for the misunderstanding, ", end='')
        elif numbers.group(1):
            needsConfirmation = True
        elif not re.search(r'(?i)\b(copies|copy|books?)\b.*?\b' + str(quant) + r'\b|\b' + str(quant) + r'\b.*?\b(copies|copy|books?)\b', prompt):
            needsConfirmation = True
        else:
            return int(quant)
        
        if needsConfirmation and not skip:
            print(f"To confirm, you'd like to order {quant} copies?")
            if confirmation():
                return int(quant)
            else:
                print("Sorry for the misunderstanding, ", end='')
    
    while True:
        print("How many copies would you like?")
        while True:
            answer, shouldRetry = handleInputWithIntents(input("Please enter your prompt (QUIT to exit) (CANCEL to cancel order): "), 'quantity')
            if shouldRetry:
                continue
            break
        if answer.lower() == "quit":
            exit()
        elif 'cancel' in answer.lower():
            return None
        numbers = re.search(reQuantityExtract, answer)
        if numbers:
            if numbers.group(1):
                quant = wordToInt(numbers.group(1))
            else:
                quant = numbers.group(2)
            
            if re.match(r'^\s*\d+\s*$', answer):
                inStock, available = stockCheck(book, int(quant))
                if not inStock:
                    print(generateContextualError('stock_insufficient', available))
                else:
                    return int(quant)
            else:
                print(f"To confirm, you'd like to order {quant} copies?")
                if confirmation():
                    inStock, available = stockCheck(book, int(quant))
                    if not inStock:
                        print(generateContextualError('stock_insufficient', available))
                    else:
                        return int(quant)

def getPickupOrDelivery(prompt: str, book: str, quantity: int, price: float) -> bool | None:
    rePickupExtract = r"(?i)\b(pick-?up|drop-?off)\b"
    reDeliveryExtract = r"(?i)\b(delivery|home)\b"
    pickupMatch = re.search(rePickupExtract, prompt)
    deliveryMatch = re.search(reDeliveryExtract, prompt)
    
    if pickupMatch:
        print("You would like to pick-up from one of our locations, right?")
        if confirmation():
            return True
    elif deliveryMatch:
        print("You would like this order to be for home delivery, right?")
        if confirmation():
            return False
    
    while True:
        print("For pickup or delivery? (type pickup/delivery)")
        while True:
            answer, shouldRetry = handleInputWithIntents(input("Please enter your prompt (QUIT to exit) (CANCEL to cancel order): "), 'pickup_delivery')
            if shouldRetry:
                continue
            break
        if answer.lower() == "quit":
            exit()
        elif 'cancel' in answer.lower():
            return None
        
        isCorrection, corrType, newValue = detectCorrection(answer)
        if isCorrection:
            if corrType == 'quantity':
                newQuantity = int(newValue)
                inStock, available = stockCheck(book, newQuantity)
                if inStock:
                    print(f"Updating quantity to {newQuantity}...")
                    quantity = newQuantity
                    price = getPrice(book) * float(quantity)
                else:
                    if available == False:
                        print(f"Sorry, '{book}' is not in stock.")
                    else:
                        print(f"Sorry, we only have {available} copies of '{book}' available.")
                    print("Please try a different quantity.")
                    continue
            elif corrType == 'book':
                matches = fuzzySearchTitle(newValue)
                if matches and matches[0][1] <= 5:
                    print(f"Updating book to {matches[0][0]}...")
                    book = matches[0][0]
                    updateContext('lastBook', book)
                    price = getPrice(book) * float(quantity)
                else:
                    print(f"Sorry, couldn't find '{newValue}'. Keeping {book}.")
            print("So, pickup or delivery?")
            continue
        
        if answer.lower().find("deliv") != -1:
            return False
        elif answer.lower().find("pick") != -1:
            return True
        else:
            print("Please specify 'pickup' or 'delivery'.")
            continue

def getLocationSelection(book: str, quantity: int, price: float) -> str | None:
    attempts = 0
    while attempts < 4:
        print("Which BlackSmith™'s store location would you like to pick-up your order from? (type 'list' to get a list of all locations)")
        while True:
            answer, shouldRetry = handleInputWithIntents(input("Please enter your prompt (QUIT to exit) (CANCEL to cancel order): "), 'location')
            if shouldRetry:
                continue
            break
        if answer.lower() == "quit":
            exit()
        elif 'cancel' in answer.lower():
            return None
        elif answer.lower().find("list") != -1:
            print("Our bookstores can be found in the following locations:")
            for location in getAllLocations():
                print(f"Location: {location[0]}, Address: {location[1]}")
        else:
            isCorrection, corrType, newValue = detectCorrection(answer)
            if isCorrection:
                if corrType == 'quantity':
                    newQuantity = int(newValue)
                    inStock, available = stockCheck(book, newQuantity)
                    if inStock:
                        print(f"Updating quantity to {newQuantity}...")
                        quantity = newQuantity
                        price = getPrice(book) * float(quantity)
                    else:
                        if available == False:
                            print(f"Sorry, '{book}' is not in stock.")
                        else:
                            print(f"Sorry, we only have {available} copies of '{book}' available.")
                        print("Please try a different quantity.")
                        continue
                elif corrType == 'book':
                    matches = fuzzySearchTitle(newValue)
                    if matches and matches[0][1] <= 5:
                        print(f"Updating book to {matches[0][0]}...")
                        book = matches[0][0]
                        updateContext('lastBook', book)
                        price = getPrice(book) * float(quantity)
                    else:
                        print(f"Sorry, couldn't find '{newValue}'. Keeping {book}.")
                print("Now, which location for pickup?")
                continue
            
            location = extractLocation(answer)
            if location:
                print(f'Okay! I have set your order to be picked up from the BlackSmith store in {location.title()}!')
                return location
            else:
                attempts += 1
                if attempts == 2:
                    print("If you are entering the address and I'm failing to recognise it, I am just looking for the general location, e.g: 'London'.")
                    print("For your reference, our bookstores can be found in the following locations:")
                    for location in getAllLocations():
                        print(f"Location: {location[0]}, Address: {location[1]}")
                if attempts > 3:
                    error = generateContextualError('location_not_found', answer)
                    print(addDiscourseMarker('clarification', f"{error}\nWould you like to cancel this order? If not, I'll keep trying."))
                    if confirmation():
                        print(addDiscourseMarker('result', "I've cancelled this order."))
                        collectFeedback(answer)
                        return None
                    else:
                        print(addDiscourseMarker('continuation', "I'll keep trying!"))
                        attempts = 0
    return None

def getDeliveryAddress(book: str, quantity: int, price: float) -> tuple[str, float] | None:
    print("What is your delivery address?")
    print("Please provide your full address in the format: House number, Street, City, Postcode")
    validAddress = False
    attempts = 0
    while not validAddress:
        while True:
            answer, shouldRetry = handleInputWithIntents(input("Please enter your prompt (QUIT to exit) (CANCEL to cancel order): "), 'general')
            if shouldRetry:
                continue
            break
        if answer.lower() == "quit":
            exit()
        elif 'cancel' in answer.lower():
            return None
        
        isCorrection, corrType, newValue = detectCorrection(answer)
        if isCorrection:
            if corrType == 'quantity':
                newQuantity = int(newValue)
                inStock, available = stockCheck(book, newQuantity)
                if inStock:
                    print(f"Updating quantity to {newQuantity}...")
                    quantity = newQuantity
                    price = getPrice(book) * float(quantity)
                    print("Now, back to your delivery address...")
                    continue
                else:
                    if available == False:
                        print(f"Sorry, '{book}' is not in stock.")
                    else:
                        print(f"Sorry, we only have {available} copies of '{book}' available.")
                    print("Please try a different quantity.")
                    continue
            elif corrType == 'book':
                matches = fuzzySearchTitle(newValue)
                if matches and matches[0][1] <= 5:
                    print(f"Updating book to {matches[0][0]}...")
                    book = matches[0][0]
                    updateContext('lastBook', book)
                    price = getPrice(book) * float(quantity)
                    print("Now, back to your delivery address...")
                    continue
        
        addressInput = answer.strip()
        if len(addressInput) < 10:
            error = generateContextualError('address_invalid')
            print(f"{error} Please provide a complete address including postcode.")
            attempts += 1
            if attempts > 3:
                print("Would you like to cancel this order? If not, I'll keep trying.")
                if confirmation():
                    print(addDiscourseMarker('result', "I've cancelled this order."))
                    collectFeedback(addressInput)
                    return None
                else:
                    print(addDiscourseMarker('continuation', "I'll keep trying!"))
                    attempts = 0
            continue
        
        postcodePattern = r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b'
        postcodeMatch = re.search(postcodePattern, addressInput, re.IGNORECASE)
        
        if not postcodeMatch:
            error = generateContextualError('address_invalid')
            print(f"{error}\nUK postcodes should look like: SW1A 1AA, M1 1AE, B33 8TH, etc.")
            attempts += 1
            if attempts > 3:
                print("Would you like to cancel this order? If not, I'll keep trying.")
                if confirmation():
                    print(addDiscourseMarker('result', "I've cancelled this order."))
                    collectFeedback(addressInput)
                    return None
                else:
                    print(addDiscourseMarker('continuation', "I'll keep trying!"))
                    attempts = 0
            continue
        
        postcode = postcodeMatch.group(0).upper()
        addressParts = addressInput.split(',')
        
        if len(addressParts) < 3:
            print("Your address should have at least: Street, City, Postcode")
            print("Please separate parts with commas, for example: 123 High Street, London, SW1A 1AA")
            attempts += 1
            if attempts > 3:
                error = generateContextualError('address_invalid')
                print(f"{error}\nWould you like to cancel this order? If not, I'll keep trying.")
                if confirmation():
                    print(addDiscourseMarker('result', "I've cancelled this order."))
                    collectFeedback(addressInput)
                    return None
                else:
                    print(addDiscourseMarker('continuation', "I'll keep trying!"))
                    attempts = 0
            continue
        
        validAddress = True

        print(f"To confirm, your delivery address is: {addressInput}?")
        print(f"Postcode detected: {postcode}")
        
        if confirmation():
            print(addDiscourseMarker('confirmation', f'Delivery will cost an additional £4.99, is that ok?'))
            if confirmation():
                return addressInput, 4.99
            else:
                print(addDiscourseMarker('result','your order has been cancelled.'))
                collectFeedback(addressInput)
                return None
        else:
            validAddress = False
            attempts += 1
            if attempts > 3:
                print(generateContextualError('address_invalid') + "\n" + \
                    "Would you like to cancel this order? If not, I'll keep trying to understand your address.")
                if confirmation():
                    print(addDiscourseMarker('result', 
                        "I've cancelled this order as I couldn't understand your delivery address."))
                    collectFeedback(addressInput)
                    return None
                else:
                    print(addDiscourseMarker('continuation', "I'll keep trying!"))
                    attempts = 0
    return None

'''
Use a slot filling approach.
Once this handler is called, enter a loop which follows the flow described below.
Flow:
1. scan the directed initial prompt for any already provided information.
2. for non-provided information, order of questions will be:
    - book 
    - quantity
    - pickup or delivery
    - for pickup:
        - ask location (check against actual locations)
        - ask pickup date and time (check against location opening hours and days)
        - confirm cost
    - for delivery:
        - ask address (check against some verification, plus provide a structure to use)
        - confirm cost (book + postage)
'''
def order(prompt: str, intents=None, count=None, tfidf=None, invIdxIntents=None, 
          qa=None, countQa=None, tfidfQa=None, invIdxQa=None,
          bookDesc=None, countBookDesc=None, tfidfBookDesc=None, invIdxBookDesc=None):
    global _intents, _count, _tfidf, _invIdxIntents
    global _qa, _countQa, _tfidfQa, _invIdxQa
    global _bookDesc, _countBookDesc, _tfidfBookDesc, _invIdxBookDesc
    _intents = intents
    _count = count
    _tfidf = tfidf
    _invIdxIntents = invIdxIntents
    _qa = qa
    _countQa = countQa
    _tfidfQa = tfidfQa
    _invIdxQa = invIdxQa
    _bookDesc = bookDesc
    _countBookDesc = countBookDesc
    _tfidfBookDesc = tfidfBookDesc
    _invIdxBookDesc = invIdxBookDesc
    
    updateContext('lastIntent', 'order')
    
    book: str = None
    quantity: int = None
    pickup: bool = None
    address: str = None
    price: float = None
    name: str = None

    book = getBookSelection(prompt)
    if not book:
        return
    
    updateContext('lastBook', book)
    
    quantity = getQuantitySelection(prompt, book)
    if not quantity:
        return
    
    if quantity and book:
        print(f"Quantity: {quantity}")
        price = getPrice(book) * float(quantity)
        
        pickup = getPickupOrDelivery(prompt, book, quantity, price)
        if pickup is None:
            return
        
        if pickup:
            address = getLocationSelection(book, quantity, price)
            if not address:
                return
            if address:
                # If the user successfully selected a pickup location, 
                # Need to select a pick-up date and timeslot.
                # Need to prevent selecting a date when the specific location is closed, 
                # or a time when the location is closed.
                result = getPickupDate(address, book, quantity, price)
                if isinstance(result, tuple):
                    date, book, quantity, price = result
                else:
                    date = result
                # -1 means the user cancelled the order in the handlers
                if date == -1:
                    return
                print(f"Okay! I've set the date for pickup to {date.day:02d}/{date.month:02d}/{date.year:02d}")
                result = getPickupTime(address, book, quantity, price)
                if isinstance(result, tuple):
                    time, book, quantity, price = result
                else:
                    time = result
                # -1 means the user cancelled the order in the handlers
                if time == -1:
                    return
        else:
            result = getDeliveryAddress(book, quantity, price)
            if not result:
                return
            address, deliveryCharge = result
            price += deliveryCharge
            print(addDiscourseMarker('confirmation', f"This charge has been added."))

    if address:
        # Check if we know the user's name, if we do not, ask for the order and save for later too.
        # If we do, just use that for the order name.
        nameSaved = readName()
        if nameSaved:
            name = nameSaved
        else:
            # Force the identity function to ask the user for their name and save it.
            identity("What is my name")
            name = readName()
            print(f"I've set the name for your order as {name}")
    
    if book and quantity and price and address and name:
        if pickup:
            storeOrder(book, getISBNbyTitle(book), quantity, pickup, address, getUnixEpochTimestamp(date.day, date.month, date.year), time, price, name)
            dateStr = date.strftime('%d/%m/%Y')
            summary = aggregateOrderDetails(book, quantity, price, 'pickup', address, dateStr, time)
            print(addDiscourseMarker('confirmation', f"Your order for {summary} has been placed!"))
            collectFeedback()
        else:
            storeOrder(book, getISBNbyTitle(book), quantity, pickup, address, None, None, price, name)
            summary = aggregateOrderDetails(book, quantity, price, 'delivery', address)
            print(addDiscourseMarker('confirmation', f"Your order for {summary} has been placed!"))
            collectFeedback()

def getPickupDate(location: str, book: str | None = None, quantity: int | None = None, price: float | None = None):
    # Dates will be based on a unix epoch timestamp for simple storage.
    date = None
    orderUpdated = False
    while True:
        print("On what date would you like to pickup from this store?\nE.g: 'tomorrow', '2-12', '2/12', '2/12/26', 'between the 12th of December and 15th of December'")
        while True:
            answer, shouldRetry = handleInputWithIntents(input("Please enter your prompt (QUIT to exit) (CANCEL to cancel order): "), 'date')
            if shouldRetry:
                continue
            break
        if answer.lower() == "quit":
            exit()
        if 'cancel' in answer.lower():
            return -1
        
        # Check for correction in the date input
        if book and quantity is not None and price is not None:
            isCorrection, corrType, newValue = detectCorrection(answer)
            if isCorrection:
                orderUpdated = True
                if corrType == 'quantity':
                    newQuantity = int(newValue)
                    from dataAccess import stockCheck as stockCheckFunc
                    inStock, available = stockCheckFunc(book, newQuantity)
                    if inStock:
                        print(f"Updating quantity to {newQuantity}...")
                        quantity = newQuantity
                        from dataAccess import getPrice as getPriceFunc
                        price = getPriceFunc(book) * float(quantity)
                    else:
                        if available == False:
                            print(f"Sorry, '{book}' is not in stock.")
                        else:
                            print(f"Sorry, we only have {available} copies of '{book}' available.")
                        print("Please try a different quantity.")
                        continue
                elif corrType == 'book':
                    matches = fuzzySearchTitle(newValue)
                    if matches and matches[0][1] <= 5:
                        print(f"Updating book to {matches[0][0]}...")
                        book = matches[0][0]
                        from context import updateContext
                        updateContext('lastBook', book)
                        from dataAccess import getPrice as getPriceFunc
                        price = getPriceFunc(book) * float(quantity)
                print("Now, what date for pickup?")
                continue
        
        monthNames = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9, 'october': 10,
            'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        
        # Check for date range FIRST (e.g., "between tomorrow and Friday")
        rangeExtract = r"(?i)between\s+(.+?)\s+and\s+(.+?)$"
        rangeResult = re.search(rangeExtract, answer)
        if rangeResult:
            startStr = rangeResult.group(1).strip()
            endStr = rangeResult.group(2).strip()
            
            startDate = None
            endDate = None
            
            relMatch1 = re.search(r"(?i)\b(today|tomorrow|day\safter\stomorrow)\b", startStr)
            relMatch2 = re.search(r"(?i)\b(today|tomorrow|day\safter\stomorrow)\b", endStr)
            
            today = datetime.date.today()
            
            if relMatch1:
                token = relMatch1.group(1).lower()
                if token == "today":
                    startDate = today
                elif token == "tomorrow":
                    startDate = today + datetime.timedelta(days=1)
                else:
                    startDate = today + datetime.timedelta(days=2)
            
            if relMatch2:
                token = relMatch2.group(1).lower()
                if token == "today":
                    endDate = today
                elif token == "tomorrow":
                    endDate = today + datetime.timedelta(days=1)
                else:
                    endDate = today + datetime.timedelta(days=2)
            
            numMatch1 = re.search(r"(?i)(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(\w+)", startStr)
            numMatch2 = re.search(r"(?i)(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(\w+)", endStr)
            
            if numMatch1 and not startDate:
                try:
                    day = int(numMatch1.group(1))
                    monthStr = numMatch1.group(2).lower()
                    if monthStr in monthNames:
                        month = monthNames[monthStr]
                        year = today.year
                        try:
                            cand = datetime.date(year, month, day)
                            if cand < today:
                                year += 1
                            startDate = datetime.date(year, month, day)
                        except ValueError:
                            pass
                except:
                    pass
            
            if numMatch2 and not endDate:
                try:
                    day = int(numMatch2.group(1))
                    monthStr = numMatch2.group(2).lower()
                    if monthStr in monthNames:
                        month = monthNames[monthStr]
                        year = today.year
                        try:
                            cand = datetime.date(year, month, day)
                            if cand < today:
                                year += 1
                            endDate = datetime.date(year, month, day)
                        except ValueError:
                            pass
                except:
                    pass
            
            if startDate and endDate and startDate <= endDate:
                candidateDates = []
                current = startDate
                while current <= endDate:
                    if current >= today:
                        open, openings = isLocOpenOnDate(location, getUnixEpochTimestamp(current.day, current.month, current.year))
                        if open:
                            candidateDates.append(current)
                    current += datetime.timedelta(days=1)
                
                if candidateDates:
                    selectedDate = candidateDates[0]
                    print(f"I found {len(candidateDates)} available date(s) in that range.")
                    print(f"The earliest available date is {selectedDate.strftime('%A, %d %B %Y')}.")
                    print("Would you like to pick up on this date?")
                    if confirmation():
                        date = selectedDate
                    else:
                        if len(candidateDates) > 1:
                            print("Here are all available dates in your range:")
                            for i, d in enumerate(candidateDates[:5], 1):
                                print(f"{i}. {d.strftime('%A, %d %B %Y')}")
                            if len(candidateDates) > 5:
                                print(f"... and {len(candidateDates) - 5} more")
                        continue
                else:
                    print(f"Sorry, the {location} location is not open on any date in that range.")
                    continue
        
        # Extractions for when the user has said a relative date. E.g: 'tomorrow', 'day after tomorrow', etc.
        relativeExtract = r"(?i)\b(?:today|day\wafter\wtomorrow|tomorrow)\b"
        relResult = re.search(relativeExtract, answer)
        if relResult and not date:
            token = relResult.group(0).lower()
            today = datetime.date.today()
            if token == "today":
                print(generateContextualError('date_invalid', 'same_day'))
                continue
            elif token == "tomorrow":
                date = today + datetime.timedelta(days=1)
            else:
                today = datetime.date.today()
                date = today + datetime.timedelta(days=2)
        
        naturalDateExtract = r"(?i)(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(\w+)(?:\s+(\d{4}))?"
        naturalResult = re.search(naturalDateExtract, answer)
        if naturalResult and not date:
            try:
                day = int(naturalResult.group(1))
                monthStr = naturalResult.group(2).lower()
                yearStr = naturalResult.group(3)
                
                if monthStr in monthNames:
                    month = monthNames[monthStr]
                    today = datetime.date.today()
                    
                    if yearStr:
                        year = int(yearStr)
                    else:
                        year = today.year
                        try:
                            cand = datetime.date(year, month, day)
                            if cand < today:
                                year += 1
                        except ValueError:
                            pass
                    
                    date = datetime.date(year, month, day)
            except Exception:
                date = None
        
        dateExtract = r"(?i)\b(?:0?[1-9]|[12][0-9]|3[01])[/-](?:0?[1-9]|1[0-2])(?:[/-](?:\d{2}|\d{4}))?\b"
        dateResult = re.search(dateExtract, answer)
        if dateResult and not date:
            dayMonth = dateResult.group(0).split("-") if '-' in dateResult.group(0) else dateResult.group(0).split('/')
            try:
                day = int(dayMonth[0])
                month = int(dayMonth[1])
                today = datetime.date.today()
                if len(dayMonth) == 3:
                    year = int(dayMonth[2])
                    if year < 100:
                        year += 2000
                else:
                    year = today.year
                    try:
                        cand = datetime.date(year, month, day)
                        if cand < today:
                            year += 1
                    except ValueError:
                        raise
                date = datetime.date(year, month, day)
            except Exception:
                date = None
        if date:
            if date < datetime.date.today():
                print(generateContextualError('date_invalid', 'past'))
                continue
            open, openings = isLocOpenOnDate(location, getUnixEpochTimestamp(date.day, date.month, date.year))
            if open:
                if orderUpdated:
                    return (date, book, quantity, price)
                return date
            else:
                print(generateContextualError('location_closed', location))
                print(f"The {location} location is not open on {date.strftime('%A %d %B %Y')}, but is open on every:")
                weekdayIter = 0
                daysOfWeek = ['Monday, ','Tuesday, ','Wednesday, ','Thursday, ','Friday, ','Saturday, ','Sunday, ']
                for char in openings:
                    # openings is a string where each weekday is 1-hot encoded for open or not.
                    if char == '1':
                        print(daysOfWeek[weekdayIter],end='')
                    weekdayIter+=1
                print('')
        else:
            print(generateContextualError('date_invalid', 'format'))
    
'''
Prompts user in a loop to get the desired time for the pickup.
Returns standardised hour of pickup in 24 hour format integer.
E.g: 1pm -> returns int 13.
If failed, returns -1.
'''
def getPickupTime(location: str, book: str | None = None, quantity: int | None = None, price: float | None = None):
    # Time will be stored in 24 hour format with no minutes.
    # If the user enters a time which is not a round hour, truncate.
    time = None
    genTimesMap = {
        'noon': 12,
        'midnight': 00,
        'lunchtime': 13,
        'morning': 10,
        'afternoon': 15,
        'evening': 18,
    }
    attempts = 0
    orderUpdated = False
    while True:
        print(f"What time would you like to pick up your order from the {location.title()} location?\nE.g: '3pm', '15:00', 'between 9am and 2pm' 'afternoon'")
        while True:
            answer, shouldRetry = handleInputWithIntents(input("Please enter your prompt (QUIT to exit) (CANCEL to cancel order): "), 'time')
            if shouldRetry:
                continue
            break
        if answer.lower() == "quit":
            exit()
        if "cancel" in answer.lower():
            return -1
        
        # Check for correction in the time input
        if book and quantity is not None and price is not None:
            isCorrection, corrType, newValue = detectCorrection(answer)
            if isCorrection:
                orderUpdated = True
                if corrType == 'quantity':
                    newQuantity = int(newValue)
                    from dataAccess import stockCheck as stockCheckFunc
                    inStock, available = stockCheckFunc(book, newQuantity)
                    if inStock:
                        print(f"Updating quantity to {newQuantity}...")
                        quantity = newQuantity
                        from dataAccess import getPrice as getPriceFunc
                        price = getPriceFunc(book) * float(quantity)
                    else:
                        if available == False:
                            print(f"Sorry, '{book}' is not in stock.")
                        else:
                            print(f"Sorry, we only have {available} copies of '{book}' available.")
                        print("Please try a different quantity.")
                        continue
                elif corrType == 'book':
                    matches = fuzzySearchTitle(newValue)
                    if matches and matches[0][1] <= 5:
                        print(f"Updating book to {matches[0][0]}...")
                        book = matches[0][0]
                        from context import updateContext
                        updateContext('lastBook', book)
                        from dataAccess import getPrice as getPriceFunc
                        price = getPriceFunc(book) * float(quantity)
                print("Now, what time for pickup?")
                continue
        
        # Check for time range first (e.g., "between 9am and 1pm")
        timeRangeExtract = r"(?i)between\s+(.+?)\s+(?:and|to)\s+(.+?)(?:\s|$)"
        timeRangeRes = re.search(timeRangeExtract, answer)
        if timeRangeRes:
            startStr = timeRangeRes.group(1).strip()
            endStr = timeRangeRes.group(2).strip()
            
            startTime = None
            endTime = None
            
            # Parse start time
            startMatch = re.search(r"(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.)", startStr, re.IGNORECASE)
            if startMatch:
                hour = int(startMatch.group(1))
                meridiem = startMatch.group(2).lower()
                if 'pm' in meridiem or 'p.m.' in meridiem:
                    startTime = 12 if hour == 12 else hour + 12
                else:
                    startTime = 0 if hour == 12 else hour
            else:
                # Check for general time or 24hr
                if startStr.lower() in genTimesMap:
                    startTime = genTimesMap[startStr.lower()]
                else:
                    match24 = re.search(r"([01]\d|2[0-3])", startStr)
                    if match24:
                        startTime = int(match24.group(1))
            
            # Parse end time
            endMatch = re.search(r"(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.)", endStr, re.IGNORECASE)
            if endMatch:
                hour = int(endMatch.group(1))
                meridiem = endMatch.group(2).lower()
                if 'pm' in meridiem or 'p.m.' in meridiem:
                    endTime = 12 if hour == 12 else hour + 12
                else:
                    endTime = 0 if hour == 12 else hour
            else:
                # Check for general time or 24hr
                if endStr.lower() in genTimesMap:
                    endTime = genTimesMap[endStr.lower()]
                else:
                    match24 = re.search(r"([01]\d|2[0-3])", endStr)
                    if match24:
                        endTime = int(match24.group(1))
            
            if startTime is not None and endTime is not None and startTime < endTime:
                # Find available times in the range
                candidateTimes = []
                for hour in range(startTime, endTime + 1):
                    isOpen, open, close = isLocOpenAtTime(location, hour)
                    if isOpen:
                        candidateTimes.append(hour)
                
                if candidateTimes:
                    time = candidateTimes[0]
                    print(f"I found {len(candidateTimes)} available time(s) in that range.")
                    print(f"The earliest available time is {time}:00 ({time if time <= 12 else time - 12}{'am' if time < 12 else 'pm'}).")
                    print("Would you like to pick up at this time?")
                    if confirmation():
                        if orderUpdated:
                            return (time, book, quantity, price)
                        return time
                    else:
                        if len(candidateTimes) > 1:
                            print("Here are all available times in your range:")
                            for i, t in enumerate(candidateTimes[:5], 1):
                                print(f"{i}. {t}:00 ({t if t <= 12 else t - 12}{'am' if t < 12 else 'pm'})")
                            if len(candidateTimes) > 5:
                                print(f"... and {len(candidateTimes) - 5} more")
                        continue
                else:
                    print(f"Sorry, the {location} location is not open during any time in that range.")
                    continue
        
        timeExtract = r"""
            (?i)                        
            \b(?:at|around|about|for|in\sthe)?\s*   # leading word
            (?:
            (\d{1,2})            # 12 hour hour
                (?::([0-5]\d))?    # optional minutes
                \s*(am|pm|a\.m\.|p\.m\.)\b
            |
            ([01]\d|2[0-3])     # 24 hour hour
                (?::([0-5]\d))\b  # minutes (allow 1430 or 14:30)
            |
            (noon|midnight|lunchtime|morning|afternoon|evening)\b # general timings
            )
        """
        timeRes = re.search(timeExtract, answer, re.VERBOSE)
        if timeRes:
            if timeRes.group(1):
                # Extracted 12 hour hour
                if timeRes.group(3):
                    hour = int(timeRes.group(1))
                    if timeRes.group(3) in ['pm','p.m.']:
                        # 12pm is 12 (noon), 1-11pm add 12
                        time = 12 if hour == 12 else hour + 12
                    else:
                        # 12am is 0 (midnight), 1-11am stay same
                        time = 0 if hour == 12 else hour
                else:
                    print(f"Is that {timeRes.group(1)} am or pm?")
                    while True:
                        answer, shouldRetry = handleInputWithIntents(input("Please enter your prompt (QUIT to exit) (CANCEL to cancel order): "), 'general')
                        if shouldRetry:
                            continue
                        break
                    if answer.lower() == "quit":
                        exit()
                    if 'cancel' in answer.lower():
                        return -1
                    hour = int(timeRes.group(1))
                    if 'pm' in answer.lower() or 'afternoon' in answer.lower():
                        time = 12 if hour == 12 else hour + 12
                    else:
                        time = 0 if hour == 12 else hour
            elif timeRes.group(4):
                # Extracted a 24 hour time, i.e it's above 12 for the hour.
                time = int(timeRes.group(4))
            elif timeRes.group(6):
                # Extracted a general timing, e.g: lunchtime.
                time = genTimesMap[timeRes.group(6).lower()]
            else:
                if attempts > 2:
                    # If above 2 attempts, suggest an available time for this location.
                    print("I was unable to understand your desired time.")
                    # Suggest 10am as a default if location is open for it.
                    suggested = 10
                    isOpen, open, close = isLocOpenAtTime(location, suggested)
                    if not isOpen:
                        suggested = open + 2
                    print(f"I can suggest {suggested} as an available time for the {location} location. Do you want to accept?")
                    if confirmation():
                        return suggested
                    else:
                        print(addDiscourseMarker('continuation', "What time would you prefer, instead?"))
                elif attempts > 3:
                    print(addDiscourseMarker('clarification', 
                        "I'm having trouble understanding you. " + \
                        "You can include 'cancel' in your response to cancel this order, or keep trying with another time."))
                else:
                    print(addDiscourseMarker('clarification', 
                        generateContextualError('time_invalid') + " " + \
                        "For example, please enter '10am'."))
                attempts += 1
        if time:
            isOpen, open, close = isLocOpenAtTime(location, time)
            if isOpen:
                if orderUpdated:
                    return (time, book, quantity, price)
                return time
            else:
                print(addDiscourseMarker('clarification', 
                    f"The {location} location is not open at {time}. " + \
                    f"It's open from {open}:00 to {close}:00."))
