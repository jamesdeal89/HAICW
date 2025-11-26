# ------ Orders/Transactions ------

import re
import datetime

from data_access import (
    fuzzySearchTitle, stockCheck, getPrice, getAllLocations, extractLocation,
    isLocOpenOnDate, isLocOpenAtTime, storeOrder, getISBNbyTitle, readName
)
from utils import confirmation, wordToInt, getUnixEpochTimestamp
from handlers import identity

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
def order(prompt: str):
    # TODO: Even in the middle of a transaction - 'how are you' should work and activate small talk intent. Must fix this.
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
    exactTitle = None
    if title:
        exactTitle = fuzzySearchTitle(title.group(1))
    if exactTitle:
        # Extracted a title from the prompt and able to fuzzy find it in the stock dataset.
        # Set the book slot to the exact title.
        print(f"Okay! So you'd like to order: {exactTitle}?")
        if confirmation():
            book = exactTitle
    elif not exactTitle:
        # Extracted a title from the prompt, but unable to find it in the stock dataset.
        print("You'd like to place an order for which book? (please enter just the title)")
        answer = input("Please enter your prompt (QUIT to exit): ")
        if answer.lower() == "quit":
            exit()
        attempts = 0
        # Allow 3 re-entry attempts, if still no match, display list of titles in stock
        while True:
            exactTitle = fuzzySearchTitle(answer)
            if exactTitle: 
                print(f"Okay! So you'd like to order: {exactTitle}?")
                if confirmation():
                    book = exactTitle
                    break
                else:
                    answer = input("Please enter your prompt (QUIT to exit): ")
                    if answer.lower() == "quit":
                        exit()
            else:
                print("Sorry that didn't match any titles in our stock database, try again")
                answer = input("Please enter your prompt (QUIT to exit): ")
                if answer.lower() == "quit":
                    exit()
            attempts += 1
            if not (attempts > 3):
                print("Sorry, we don't stock that book.")
                break

    skip = False
    if book:
        if numbers:
            # Detected a numerical value in the user's prompt.
            # Confirm with user if this is the quantity they want to order.
            
            # Need to convert word for number into number.
            if numbers.group(1):
                quant = wordToInt(numbers.group(1).lower())
            else:
                quant = numbers.group(2)

            print(f"To confirm, you'd like to order {quant} copies?")
            if confirmation():
                quantity = quant
                skip = True
            else:
                print("Sorry for the misunderstanding, ", end='')
        if not skip:
            while True:
                print("How many copies would you like?")
                answer = input("Please enter your prompt (QUIT to exit) (CANCEL to cancel order): ")
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
                    print(f"To confirm, you'd like to order {quant} copies?")
                    if confirmation():
                        # Perform a stock check
                        inStock, available = stockCheck(book, int(quant))
                        if not inStock:
                            print(f"Unfortunately, we don't have {quant} available. There are only {available} copies in stock.")
                        else:
                            quantity = int(quant)
                            break
    
    if quantity and book:
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
            print("For pickup or delivery? (type pickup/delivery)")
            answer = input("Please enter your prompt (QUIT to exit): ")
            if answer.lower() == "quit":
                exit()
            elif answer.find("delivery") != -1:
                pickup = False
            else:
                pickup = True
        # Based on pickup boolean, get home address details or get store location.
        if pickup:
            attempts = 0
            while attempts < 4:
                print("Which BlackSmith™'s store location would you like to pick-up your order from? (type 'list' to get a list of all locations)")
                answer = input("Please enter your prompt (QUIT to exit): ")
                if answer.lower() == "quit":
                    exit()
                elif answer.lower().find("list") != -1:
                    # Help the user discover what locations exist.
                    print("Our bookstores can be found in the following locations:")
                    for location in getAllLocations():
                        print(f"Location: {location[0]}, Address: {location[1]}")
                else:
                    # Search for the location they specified.
                    # After 3 fails to recognise a location name, print out the list of locations even if the user didn't ask.
                    location = extractLocation(answer)
                    if location:
                        print(f'Okay! I have set your order to be picked up from the BlackSmith store in {location}!')
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
                            print("I'm sorry I'm unable to understand which location you'd like to pickup from.\n" \
                            "Would you like to cancel this order? If not, I'll keep trying to understand which location you'd prefer.")
                            if confirmation():
                                print("Okay! I'll keep trying!")
                                attempts = 0
                            else:
                                print("Okay, I'm sorry I failed to understand your desired location. \nI have cancelled this order.")
                                # TODO: Feedback mechanism - ask for user feedback (rating + open feedback) and store for developer use.
                                break
            if address:
                # If the user successfully selected a pickup location, 
                # Need to select a pick-up date and timeslot.
                # Need to prevent selecting a date when the specific location is closed, 
                # or a time when the location is closed.
                date = getPickupDate(address)
                print(f"Okay! I've set the date for pickup to {date.day:02d}/{date.month:02d}/{date.year:02d}")
                # -1 means the user cancelled the order in the handlers
                if date == -1:
                    return
                time = getPickupTime(address)
                # -1 means the user cancelled the order in the handlers
                if time == -1:
                    return
        else:
            pass

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
    
    if book and pickup and quantity and price and address and name:
        storeOrder(book, getISBNbyTitle(book), quantity, pickup, address, getUnixEpochTimestamp(date.day, date.month, date.year), time, price, name)

def getPickupDate(location: str):
    # Dates will be based on a unix epoch timestamp for simple storage.
    date = None
    while True:
        print("On what date would you like to pickup from this store?")
        answer = input("Please enter your prompt (QUIT to exit) (CANCEL to cancel order): ")
        if answer.lower() == "quit":
            exit()
        if 'cancel' in answer.lower():
            return -1
        # Extractions for when the user has said a relative date. E.g: 'tomorrow', 'day after tomorrow', etc.
        relativeExtract = r"(?i)\b(?:today|day\wafter\wtomorrow|tomorrow)\b"
        relResult = re.search(relativeExtract, answer)
        if relResult:
            token = relResult.group(0).lower()
            today = datetime.date.today()
            if token == "today":
                print("I'm sorry, but we do not support same-day pickup. Please select a future date.")
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
                print("Date cannot be in the past. Please select a future date.")
                continue
            open, openings = isLocOpenOnDate(location, getUnixEpochTimestamp(date.day, date.month, date.year))
            if open:
                return date
            else:
                print("Please select a different date, the location you chose is not open on that date.")
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
            print("I was unable to recognise which date you intended, please try again.")
    
'''
Prompts user in a loop to get the desired time for the pickup.
Returns standardised hour of pickup in 24 hour format integer.
E.g: 1pm -> returns int 13.
If failed, returns -1.
'''
def getPickupTime(location):
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
    while True:
        print(f"What time would you like to pick up your order from the {location} location?")
        answer = input("Please enter your prompt (QUIT to exit) (CANCEL to cancel order): ")
        if answer.lower() == "quit":
            exit()
        if "cancel" in answer.lower():
            return -1
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
        timeRes = re.search(answer,timeExtract)
        if timeRes:
            if timeRes.group(1):
                # Extracted 12 hour hour
                if timeRes.group(1):
                    if timeRes.groups(3) in ['pm','p.m.']:
                        time = int(timeRes.group(1) + 12)
                    else:
                        # Assume it's AM.
                        time = int(timeRes.group(1))
                else:
                    print(f"Is that {timeRes.groups(1)} am or pm?")
                    answer = input("Please enter your prompt (QUIT to exit): ")
                    if answer.lower() == "quit":
                        exit()
                    if 'pm' in answer.lower() or 'afternoon' in answer.lower():
                        time = int(timeRes.group(1)) + 12
                    else:
                        time = int(timeRes.group(1))
            elif timeRes.group(4):
                # Extracted a 24 hour time, i.e it's above 12 for the hour.
                time = int(timeRes.groups(4))
            elif timeRes.group(6):
                # Extracted a general timing, e.g: lunchtime.
                time = genTimesMap[timeRes.groups(6)]
            else:
                if attempts > 2:
                    # If above 2 attempts, suggest an available time for this location.
                    print("I was unable to understand your desired time.")
                    # Suggest 10am as a default if location is open for it.
                    suggested = 10
                    isOpen, open, close = isLocOpenAtTime(location, suggested)
                    if not isOpen:
                        # Suggest 2 hours after known opening time if not open at 10.
                        suggested = open + 2
                    print("I can suggest {suggested} as an available time for the {location} location. Do you want to accept?")
                    if confirmation():
                        return suggested
                    else:
                        print("What time would you prefer, instead?")
                elif attempts > 3:
                    # If above 3 attempts, suggest cancelling the order to give the user a way to escape the transaction flow.
                    print("I apologise, I was unable to understand you. \n" \
                    "You can include 'cancel' in your response to cancel this order, or you can keep trying and enter another time.")
                else:
                    # Encourage the user to use a simple time format which system is able to extract.
                    print("I'm sorry, I dont understand what time you'd like.\n" \
                    "For example, please enter '10am'.")
                attempts += 1
        if time:
            isOpen, open, close = isLocOpenAtTime(location, time)
            if isOpen:
                return time
            else:
                print(f"I apologise, the {location} location is not open at {time}.\n" \
                        f"It is open from {open}:00 to {close}:00.")
