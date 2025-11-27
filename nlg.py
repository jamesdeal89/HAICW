import random

bookRefs = {
    'first': lambda title: title,
    'subsequent': [
        lambda title: 'it',
        lambda title: 'this book',
        lambda title: 'this title',
        lambda title: 'the book',
        lambda title: 'this one'
    ]
}

def getReferringExpression(entity, entityType, isFirstMention):
    if entityType == 'book':
        if isFirstMention:
            return entity
        else:
            return random.choice(bookRefs['subsequent'])(entity)
    return entity

def aggregateOrderDetails(title, quantity, cost, deliveryType, location, date=None, time=None):
    parts = []
    
    if quantity > 1:
        parts.append(f"{quantity} copies of '{title}'")
    else:
        parts.append(f"'{title}'")
    
    parts.append(f"totaling £{cost:.2f}")
    
    if deliveryType == 'pickup':
        parts.append(f"for pickup at {location}")
        if date and time is not None:
            parts.append(f"on {date} at {time:02d}:00")
        elif date:
            parts.append(f"on {date}")
        elif time is not None:
            parts.append(f"at {time:02d}:00")
    else:
        parts.append(f"for delivery to {location}")
    
    if len(parts) == 2:
        return f"{parts[0]} {parts[1]}"
    else:
        return ' '.join(parts)

def generateContextualError(errorType, context=None):
    errors = {
        'book_not_found': [
            f"I couldn't find '{context}' in our stock.",
            f"Unfortunately, we don't have '{context}' available.",
            f"'{context}' isn't in our current inventory."
        ],
        'invalid_quantity': [
            "The quantity needs to be a positive number.",
            "Please specify a valid quantity.",
            "I didn't understand that quantity."
        ],
        'date_invalid': [
            "That date doesn't seem valid." if context == 'format' else 
            "We don't support same-day pickup. Please select a future date." if context == 'same_day' else
            "Date cannot be in the past. Please select a future date." if context == 'past' else
            "I couldn't understand that date format.",
            
            "Please provide a valid date." if context == 'format' else
            "Please choose a date from tomorrow onwards." if context == 'same_day' else
            "That date has already passed. Try a future date." if context == 'past' else
            "I'm having trouble with that date.",
            
            "I didn't recognize that date format." if context == 'format' else
            "Same-day orders aren't available. Select a later date." if context == 'same_day' else
            "Please pick a date in the future." if context == 'past' else
            "That date isn't working for me."
        ],
        'time_invalid': [
            "I couldn't understand what time you meant.",
            "That time doesn't seem valid.",
            "Please specify a time in a format like '10am' or '14:00'."
        ],
        'location_closed': [
            f"The {context} location is closed on that date." if context else "That location is closed on that date.",
            f"Unfortunately, {context} isn't open then." if context else "That location isn't open then.",
            f"{context} is closed that day." if context else "We're closed that day."
        ],
        'stock_insufficient': [
            f"We only have {context} copies available.",
            f"Unfortunately, we don't have enough stock. We have {context} left.",
            f"We can only fulfill an order for {context} copies right now."
        ],
        'location_not_found': [
            f"I couldn't find a location matching '{context}'.",
            f"'{context}' doesn't match any of our stores.",
            "That location doesn't seem to exist in our system."
        ],
        'address_invalid': [
            "That address doesn't look quite right.",
            "Please provide a valid UK address with a postcode.",
            "I need a complete address including a postcode."
        ],
        'generic': [
            "I'm not sure I understood that correctly.",
            "Could you try rephrasing that?",
            "I didn't quite catch that."
        ]
    }
    
    candidates = errors.get(errorType, errors['generic'])
    return random.choice(candidates)

def generateSuggestion(suggestionType, options):
    if suggestionType == 'similar_books':
        intro = random.choice([
            "Did you mean one of these?",
            "Here are some similar titles:",
            "Perhaps you're looking for:",
            "I found these similar books:"
        ])
        bookList = '\n'.join([f"  - '{book['name']}' by {book['author']}" for book in options[:3]])
        return f"{intro}\n{bookList}"
    
    elif suggestionType == 'similar_locations':
        intro = random.choice([
            "Did you mean:",
            "Here are the closest matches:",
            "Perhaps you meant:"
        ])
        locationList = '\n'.join([f"  - {loc['name']}" for loc in options[:3]])
        return f"{intro}\n{locationList}"
    
    elif suggestionType == 'available_genres':
        intro = random.choice([
            "We have books in these genres:",
            "Available genres include:",
            "You can choose from:"
        ])
        return f"{intro} {', '.join(options)}"
    
    return ""

def addDiscourseMarker(context, message):
    markers = {
        'clarification': ['Actually, ', 'To clarify, ', 'Let me explain: ', ''],
        'continuation': ['Also, ', 'Additionally, ', 'By the way, ', ''],
        'result': ['So ', 'Therefore, ', 'As a result, ', ''],
        'contrast': ['However, ', 'On the other hand, ', 'But ', ''],
        'confirmation': ['Great! ', 'Perfect! ', 'Excellent! ', 'Understood. ']
    }
    
    candidates = markers.get(context, [''])
    marker = random.choice(candidates)
    
    return f"{marker}{message}" if marker else message
