# ------ Orders/Transactions ------

import re
import datetime

from dataAccess import (
    fuzzySearchTitle, stockCheck, getPrice, getAllLocations, extractLocation,
    isLocOpenOnDate, isLocOpenAtTime, storeOrder, getISBNbyTitle, readName, storeFeedback, getLocationsAvailable
)
from utils import confirmation, wordToInt, getUnixEpochTimestamp
from handlers import identity, small, discover, thank
from search import searchIntent, question
from nlg import aggregateOrderDetails, generateContextualError, addDiscourseMarker
from context import updateContext

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
    
    # Check if input seems transaction-relevant based on context
    isTransactionRelevant = False
    
    if expectedType == 'quantity':
        # Check if it contains numbers or number words
        if re.search(r'\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve', userInput, re.IGNORECASE):
            isTransactionRelevant = True
    elif expectedType == 'book':
        # If it's a reasonable length for a book title (more than 2 chars)
        if len(userInput.strip()) > 2 and not re.match(r'^(yes|no|yeah|nope|yep)$', userInput, re.IGNORECASE):
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
        # For "order" intent, don't interrupt - treat as normal input (avoid nested orders)
    
    # Didn't match any special intent, return for normal processing
    return userInput, False

def detectCorrection(userInput):
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

def collectFeedback(lastPrompt=""):
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
          qa=None, countQa=None, tfidfQa=None, invIdxQa=None):
    global _intents, _count, _tfidf, _invIdxIntents
    global _qa, _countQa, _tfidfQa, _invIdxQa
    _intents = intents
    _count = count
    _tfidf = tfidf
    _invIdxIntents = invIdxIntents
    _qa = qa
    _countQa = countQa
    _tfidfQa = tfidfQa
    _invIdxQa = invIdxQa
    
    updateContext('lastIntent', 'order')
    
    # Declare slots as None for now, any left as None after initial scan of prompt will be ask for
    book: str = None
    quantity: int = None
    pickup: bool = None
    address: str = None
    price: float = None
    name: str = None

    # Extract the book title they want to order.
    reTitleExtract = r"(?i)\b(?:order|buy|get|purchase|place)\b.*?\b([A-Za-z0-9'':,&() ]{3,}?)\b(?=(?:\s+(?:for|from|to|at|in|pickup|delivery|delivered|store)\b|[.?!,;]|$))"
    title = re.search(reTitleExtract, prompt)
    # Extract the quantity they want to order.
    # Matches both numerical words and digits.
    reQuantityExtract = r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen| \
                        sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)\b|\b(\d+)\b"
    numbers = re.search(reQuantityExtract, prompt)
    # Extract if it's for pickup for delivery.
    rePickupExtract = r"(?i)\b(pick-?up|drop-?off)\b"
    reDeliveryExtract = r"(?i)\b(delivery|home)\b"
    pickupMatch = re.search(rePickupExtract, prompt)
    deliveryMatch = re.search(reDeliveryExtract, prompt)

    # Check for book in stock.json 
    # Use Levenshtein-based fuzzy search to ensure detected book title can match 
    # to a specific title as per stock.json.
    matches = []
    if title:
        matches = fuzzySearchTitle(title.group(1))
    if matches:
        # Extracted a title from the prompt and able to fuzzy find it in the stock dataset.
        # If distance is very low (<=2), high confidence - no need to confirm
        if matches[0][1] <= 2:
            book = matches[0][0]
            updateContext('lastBook', book)
            print(f"Ordering: {book}")
        else:
            # Lower confidence, confirm with user
            print(f"Okay! So you'd like to order: {matches[0][0]}?")
            if confirmation():
                book = matches[0][0]
            elif len(matches) > 1:
                # User said no to first match, try second match
                print(f"Did you mean: {matches[1][0]}?")
                if confirmation():
                    book = matches[1][0]
                elif len(matches) > 2:
                    # Try third match
                    print(f"Or perhaps: {matches[2][0]}?")
                    if confirmation():
                        book = matches[2][0]
    if not matches or not book:
        # Extracted a title from the prompt, but unable to find it in the stock dataset.
        print("You'd like to place an order for which book? (please enter just the title)")
        while True:
            answer, shouldRetry = handleInputWithIntents(input("Please enter your prompt (QUIT to exit) (CANCEL to cancel order): "), 'book')
            if shouldRetry:
                continue  # Intent was handled, ask again
            if answer.lower() == "quit":
                exit()
            if 'cancel' in answer.lower():
                return
            break
        attempts = 0
        # Allow 3 re-entry attempts, if still no match, display list of titles in stock
        while True:
            matches = fuzzySearchTitle(answer)
            if matches:
                # Try matches in order with confidence-based approach
                if matches[0][1] <= 2:
                    # High confidence match
                    book = matches[0][0]
                    print(f"Ordering: {book}")
                    break
                else:
                    # Ask for confirmation on first match
                    print(f"Okay! So you'd like to order: {matches[0][0]}?")
                    if confirmation():
                        book = matches[0][0]
                        break
                    elif len(matches) > 1:
                        # Try second match
                        print(f"Did you mean: {matches[1][0]}?")
                        if confirmation():
                            book = matches[1][0]
                            break
                        elif len(matches) > 2:
                            # Try third match
                            print(f"Or perhaps: {matches[2][0]}?")
                            if confirmation():
                                book = matches[2][0]
                                break
                    # None of the matches were confirmed
                    while True:
                        answer, shouldRetry = handleInputWithIntents(input("Please enter your prompt (QUIT to exit) (CANCEL to cancel order): "), 'book')
                        if shouldRetry:
                            continue
                        if answer.lower() == "quit":
                            exit()
                        if 'cancel' in answer.lower():
                            return
                        break
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
                        return
                    break
            attempts += 1
            if attempts > 3:
                print("Sorry, we don't stock that book.")
                break

    skip = False
    if book:
        if numbers:
            # Detected a numerical value in the user's prompt.
            # Need to convert word for number into number.
            if numbers.group(1):
                quant = wordToInt(numbers.group(1).lower())
            else:
                quant = numbers.group(2)
            
            # Find all numbers in the prompt to check if there are multiple
            allNumbers = re.findall(reQuantityExtract, prompt)
            
            # Only confirm if:
            # 1. There are multiple numbers (ambiguous which is quantity)
            # 2. The number is a word (less explicit than digit)
            # 3. The prompt doesn't clearly indicate quantity context
            needsConfirmation = False
            
            if len(allNumbers) > 1:
                # Multiple numbers found - iterate through them
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
                        quantity = quant
                        skip = True
                except ValueError:
                    print("Sorry for the misunderstanding, ", end='')
            elif numbers.group(1):
                # Number was a word - confirm for clarity
                needsConfirmation = True
            elif not re.search(r'(?i)\b(copies|copy|books?)\b.*?\b' + str(quant) + r'\b|\b' + str(quant) + r'\b.*?\b(copies|copy|books?)\b', prompt):
                # Number found but not in clear quantity context
                needsConfirmation = True
            else:
                # Direct numerical input in proper context, no confirmation needed
                quantity = int(quant)
                skip = True
            
            if needsConfirmation and not skip:
                print(f"To confirm, you'd like to order {quant} copies?")
                if confirmation():
                    quantity = quant
                    skip = True
                else:
                    print("Sorry for the misunderstanding, ", end='')
        
        if not skip:
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
                    return
                numbers = re.search(reQuantityExtract, answer)
                if numbers:
                    if numbers.group(1):
                        quant = wordToInt(numbers.group(1))
                    else:
                        quant = numbers.group(2)
                    
                    # Check if answer is just a number 
                    if re.match(r'^\s*\d+\s*$', answer):
                        # Direct numerical response, no confirmation needed
                        inStock, available = stockCheck(book, int(quant))
                        if not inStock:
                            print(generateContextualError('stock_insufficient', available))
                        else:
                            quantity = int(quant)
                            break
                    else:
                        # More complex answer, needs confirmation
                        print(f"To confirm, you'd like to order {quant} copies?")
                        if confirmation():
                            # Perform a stock check
                            inStock, available = stockCheck(book, int(quant))
                            if not inStock:
                                print(generateContextualError('stock_insufficient', available))
                            else:
                                quantity = int(quant)
                                break
    
    if quantity and book:
        print(f"Quantity: {quantity}")
        # Now have all the information needed to set the price for this order.
        price = getPrice(book) * float(quantity)
        if pickupMatch:
            # Keyword for pickup detected, but need to confirm.
            print("You would like to pick-up from one of our locations, right?")
            if confirmation():
                pickup = True
        elif deliveryMatch:
            # Keyword for delivery detected, but need to confirm.
            print("You would like this order to be for home delivery, right?")
            if confirmation():
                pickup = False
        else:
            # No decision is clear in the initial prompt, ask directly
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
                    return
                
                # Check for correction in the pickup/delivery response
                isCorrection, corrType, newValue = detectCorrection(answer)
                if isCorrection:
                    if corrType == 'quantity':
                        print(f"Updating quantity to {newValue}...")
                        quantity = int(newValue)
                        price = getPrice(book) * float(quantity)
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
                
                if answer.find("deliv") != -1:
                    pickup = False
                    break
                elif answer.find("pick") != -1:
                    pickup = True
                    break
                else:
                    print("Please specify 'pickup' or 'delivery'.")
                    continue
        # Based on pickup boolean, get home address details or get store location.
        if pickup:
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
                    return
                elif answer.lower().find("list") != -1:
                    # Help the user discover what locations exist.
                    print("Our bookstores can be found in the following locations:")
                    for location in getAllLocations():
                        print(f"Location: {location[0]}, Address: {location[1]}")
                else:
                    # Search for the location they specified.
                    # After 3 fails to recognise a location name, print out the list of locations even if the user didn't ask.
                    
                    # Check for correction in the location response
                    isCorrection, corrType, newValue = detectCorrection(answer)
                    if isCorrection:
                        if corrType == 'quantity':
                            print(f"Updating quantity to {newValue}...")
                            quantity = int(newValue)
                            price = getPrice(book) * float(quantity)
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
                        address = location
                        break
                    else:
                        attempts += 1
                        if attempts == 2:
                            # Help the user discover what locations exist.
                            # Elaborate on the expected format of the response from the user.
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
                                break
                            else:
                                print(addDiscourseMarker('continuation', "I'll keep trying!"))
                                attempts = 0
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
            print("What is your delivery address?")
            print("Please provide your full address in the format: House number, Street, City, Postcode")
            # Encourage the user to follow a specific format of address to give the extraction the best chances of working.
            validAddress = False
            attempts = 0
            # Keep trying until we get a valid extracted address.
            while not validAddress:
                while True:
                    answer, shouldRetry = handleInputWithIntents(input("Please enter your prompt (QUIT to exit) (CANCEL to cancel order): "), 'general')
                    if shouldRetry:
                        continue
                    break
                if answer.lower() == "quit":
                    exit()
                elif 'cancel' in answer.lower():
                    return
                
                # Check for correction in the address input
                isCorrection, corrType, newValue = detectCorrection(answer)
                if isCorrection:
                    if corrType == 'quantity':
                        print(f"Updating quantity to {newValue}...")
                        quantity = int(newValue)
                        price = getPrice(book) * float(quantity)
                        print("Now, back to your delivery address...")
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
                # Simple length check to filter out clearly incorrect responses. 
                if len(addressInput) < 10:
                    error = generateContextualError('address_invalid')
                    print(f"{error} Please provide a complete address including postcode.")
                    attempts += 1
                    if attempts > 3:
                        print("Would you like to cancel this order? If not, I'll keep trying.")
                        if confirmation():
                            print(addDiscourseMarker('result', "I've cancelled this order."))
                            collectFeedback(addressInput)
                            return
                        else:
                            print(addDiscourseMarker('continuation', "I'll keep trying!"))
                            attempts = 0
                    continue
                
                # Basic UK postcode verification.
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
                            return
                        else:
                            print(addDiscourseMarker('continuation', "I'll keep trying!"))
                            attempts = 0
                    continue
                
                postcode = postcodeMatch.group(0).upper()
                addressParts = addressInput.split(',')
                
                # Ensure the user inputs all 3 parts of the expected address.
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
                            return
                        else:
                            print(addDiscourseMarker('continuation', "I'll keep trying!"))
                            attempts = 0
                    continue
                
                validAddress = True

                # Addresses are very sensitive and consequences of getting it wrong are them losing their order, 
                # Therefore verify before setting slot.
                print(f"To confirm, your delivery address is: {addressInput}?")
                print(f"Postcode detected: {postcode}")
                
                if confirmation():
                    address = addressInput
                    print(addDiscourseMarker('confirmation', f'Delivery will cost an additional £4.99, is that ok?'))
                    if confirmation():
                        price += 4.99
                        print(addDiscourseMarker('confirmation', f"This charge has been added."))
                    else:
                        print(addDiscourseMarker('result','your order has been cancelled.'))
                        collectFeedback(addressInput)
                        return
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
                            return
                        else:
                            print(addDiscourseMarker('continuation', "I'll keep trying!"))
                            attempts = 0

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

def getPickupDate(location: str, book=None, quantity=None, price=None):
    # Dates will be based on a unix epoch timestamp for simple storage.
    date = None
    orderUpdated = False
    while True:
        print("On what date would you like to pickup from this store?")
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
                    print(f"Updating quantity to {newValue}...")
                    quantity = int(newValue)
                    from dataAccess import getPrice as getPriceFunc
                    price = getPriceFunc(book) * float(quantity)
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
        # Extractions for when the user has said a relative date. E.g: 'tomorrow', 'day after tomorrow', etc.
        relativeExtract = r"(?i)\b(?:today|day\wafter\wtomorrow|tomorrow)\b"
        relResult = re.search(relativeExtract, answer)
        if relResult:
            token = relResult.group(0).lower()
            today = datetime.date.today()
            if token == "today":
                print(generateContextualError('date_invalid', 'same_day'))
                continue
            elif token == "tomorrow":
                date = today + datetime.timedelta(days=1)
            else:
                # Set to day after tomorrow
                today = datetime.date.today()
                date = today + datetime.timedelta(days=2)
        
        # Extractions for when the user enters a date in formats:
        #   5/9, 05/09, etc
        #   06/07/2025, 05-09-25, etc.
        dateExtract = r"(?i)\b(?:0?[1-9]|[12][0-9]|3[01])[/-](?:0?[1-9]|1[0-2])(?:[/-](?:\d{2}|\d{4}))?\b"
        dateResult = re.search(dateExtract, answer)
        if dateResult:
            # Split into day and month independently.
            dayMonth = dateResult.group(0).split("-") if '-' in dateResult.group(0) else dateResult.group(0).split('/')
            try:
                day = int(dayMonth[0])
                month = int(dayMonth[1])
                today = datetime.date.today()
                if len(dayMonth) == 3:
                    # means the user included a year
                    year = int(dayMonth[2])
                    if year < 100:
                        # to account for when they truncate the year to 2 digits.
                        year += 2000
                else:
                    year = today.year
                    try:
                        # if no year was provided and the date desired has already passed, assume next year
                        cand = datetime.date(year, month, day)
                        if cand < today:
                            year += 1
                    except ValueError:
                        # when invalid day or month
                        # raise this error so the outer except catches it
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
                print(f"The {location} location is not open on {date.strftime("%A %d %B %Y")}, but is open on every:")
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
def getPickupTime(location, book=None, quantity=None, price=None):
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
        print(f"What time would you like to pick up your order from the {location} location?")
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
                    print(f"Updating quantity to {newValue}...")
                    quantity = int(newValue)
                    from dataAccess import getPrice as getPriceFunc
                    price = getPriceFunc(book) * float(quantity)
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
