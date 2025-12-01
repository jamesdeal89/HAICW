import re

sessionContext = {
    'lastBook': None,
    'lastLocation': None,
    'lastIntent': None,
    'lastGenre': None,
    'lastQuantity': None,
    'conversationTurn': 0
}

def updateContext(key, value):
    sessionContext[key] = value
    sessionContext['conversationTurn'] += 1

def getContext(key):
    return sessionContext.get(key)

def resetContext():
    global sessionContext
    sessionContext = {
        'lastBook': None,
        'lastLocation': None,
        'lastIntent': None,
        'lastGenre': None,
        'lastQuantity': None,
        'conversationTurn': 0
    }

def resolveEllipsis(query):
    queryLower = query.lower().strip()
    
    if re.match(r'^\d+$', queryLower):
        if sessionContext['lastIntent'] == 'order' and sessionContext['lastBook']:
            return f"order {sessionContext['lastBook']} {queryLower} copies"
        return query
    
    howAboutMatch = re.match(r'^(how about|what about|and)\s+(.+)', queryLower)
    if howAboutMatch:
        newEntity = howAboutMatch.group(2).strip()
        
        if sessionContext['lastIntent'] == 'check':
            if sessionContext['lastBook']:
                return f"check {newEntity} availability"
            return f"check {newEntity}"
        elif sessionContext['lastIntent'] == 'recommend':
            return f"recommend {newEntity} books"
        elif sessionContext['lastIntent'] == 'order':
            return f"order {newEntity}"
    
    if re.match(r'^(that one|it|this|that|the book)$', queryLower):
        if sessionContext['lastBook']:
            if sessionContext['lastIntent'] == 'order':
                return f"order {sessionContext['lastBook']}"
            elif sessionContext['lastIntent'] == 'check':
                return f"check {sessionContext['lastBook']} availability"
        return query
    
    pronounInPhrase = re.search(r'\b(it|that|this)\b', queryLower)
    if pronounInPhrase:
        if sessionContext['lastIntent'] in ['opening', 'address', 'facilities'] and sessionContext['lastLocation']:
            replaced = re.sub(r'\b(it|that|this)\b', sessionContext['lastLocation'], queryLower, count=1)
            return replaced
        elif sessionContext['lastBook']:
            replaced = re.sub(r'\b(it|that|this)\b', sessionContext['lastBook'], queryLower, count=1)
            return replaced
    
    locationPattern = r'^(at|in|from)\s+(.+)'
    locationMatch = re.match(locationPattern, queryLower)
    if locationMatch:
        location = locationMatch.group(2).strip()
        if sessionContext['lastIntent'] == 'check' and sessionContext['lastBook']:
            return f"check {sessionContext['lastBook']} at {location}"
        return query
    
    return query
