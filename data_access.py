import json
import os
import re
import datetime
from typing import Tuple
from nltk.corpus import stopwords
from nltk import download

download('stopwords', quiet=True)

from utils import levenshteinDistance

def readName():
    # Read name from session JSON on disk
    if os.path.exists('session.json'):
        with open("session.json", "r") as f:
            session = json.load(f)
        return session['name']

def saveName(name):
    # Save the user details into a JSON on disk 
    # Check if a JSON exists already
    if os.path.exists('session.json'):
        # Exists already, so read it, update the name field, write back.
        with open("session.json", "r") as f:
            session = json.load(f)
        session['name'] = name
        with open("session.json", "w") as f:
            json.dump(session, f)
    else:
        session = {
            "name": name
        }
        with open("session.json", "w") as f:
            json.dump(session, f)

def resetSession():
    # Delete session JSON on disk
    os.remove("session.json")

def getStockJSON():
    with open('stock.json', 'r') as f:
        data = json.load(f)
    return data

def getOrdersJSON():
    if os.path.exists('orders.json'):
        with open('orders.json', 'r') as f:
            data = json.load(f)
        return data
    return None

'''
Returns the JSON data for bookstore locations.
'''
def getLocationsJSON():
    with open('locations.json', 'r') as f:
        data = json.load(f)
    return data

'''
Returns a list of all location names and addresses in the locations.json dataset.
'''
def getAllLocations() -> list[list[str]]:
    locations = getLocationsJSON()
    allLocs = []
    for location in locations['locations']:
        loc = []
        loc.append(location['name'])
        loc.append(location['address'])
        allLocs.append(loc)
    return allLocs

'''
Attempts to extract a location from the prompt which matches one in the locations.json dataset.
If fail to extract or match, returns the empty string.
'''
def extractLocation(prompt: str) -> str:
    locations = getAllLocations()
    # Use a map to just have a 1-D list of locations, removing address.
    locations = list(map(lambda x: x[0].lower(), locations))

    # Tokenise into words and lowercase.
    tokens = re.findall(r'\b\w+\b', prompt.lower())
    stopWords = stopwords.words('english')
    # Remove all stopwords from the tokenized prompt.
    filteredTokens = [t for t in tokens if t not in stopWords]

    if not filteredTokens:
        # Means empty prompt or all stop words
        # As a fallback, attempt to directly match locations in the raw, unfiltered prompt
        promptLower = prompt.lower()
        for loc in locations:
            if re.search(r'\b' + re.escape(loc) + r'\b', promptLower):
                return loc
        return ''
    
    # Maximum number of words in any stored location name
    maxLenLoc = max(len(loc) for loc in locations)
    # Try the longest n-grams first for multiple word location names 
    for n in range(maxLenLoc, 0, -1):
        for i in range(len(filteredTokens) - n + 1):
            gram = ' '.join(filteredTokens[i:i+n])
            if gram in locations:
                return gram
    return ''

'''
Get all book JSONs of books in stock.json dataset.
'''
def getGenreBooks(genre: str) -> list:
    stock = getStockJSON()
    books = []
    for book in stock['stock']:
        if book['genre'] == genre:
            books.append(book)
    return books

'''
Get all genres of books in stock.json dataset.
'''
def getGenres() -> list[str]:
    stock = getStockJSON()
    genres = set()
    for book in stock['stock']:
        genres.add(book['genre'])
    return genres

'''
Check if desired stock amount is possible to order.
Returns bool and either: 
    -1 if True as exact number not needed by the caller in this case, 
or: 
    the integer number in stock if False, as caller needs to inform user.
'''
def stockCheck(title: str, quantity: int) -> Tuple[bool,int]:
    titleNorm = title.lower().strip()
    for book in getStockJSON()['stock']:
        if book['name'].lower().strip() == titleNorm:
            if book['count'] >= quantity:
                return True, -1
            else:
                return False, book['count']
    return False

'''
Returns the float price for a book based on it's title.
If name is not found, returns -1
'''
def getPrice(title: str) -> float:
    titleNorm = title.lower().strip()
    for book in getStockJSON()['stock']:
        if book['name'].lower().strip() == titleNorm:
            return book['price']
    return -1

'''
Perform a fuzzy search for title using Levenshtein distance.
Returns the true title per stock.json, or empty string if not found.
'''
def fuzzySearchTitle(title: str) -> str:
    titleNorm = title.lower().strip()
    levenshteinDistances = []
    for book in getStockJSON()['stock']:
        bookNorm = book['name'].lower().strip()
        # Calculate distance between user title and the stored book title
        levenshteinDistances.append((book['name'],levenshteinDistance(bookNorm, titleNorm, len(bookNorm), len(titleNorm))))
    # Sort ascending, first index will be the most similar / least distance
    levenshteinDistances.sort(key=lambda entry: entry[1])
    # TODO: return both the title and the distance, allows for confidence based checks, if low dist, no need to confirm. Also return top 3 so if they do not confirm, iterate.
    # Tolerance is a levenshtein distance of 1/3rd of the user's desired title
    if levenshteinDistances and levenshteinDistances[0][1] < len(title)//3:
        return levenshteinDistances[0][0]
    else:
        return ""

def getISBNbyTitle(title: str) -> str:
    titleNorm = title.lower().strip()
    for book in getStockJSON()['stock']:
        if book['name'].lower().strip() == titleNorm:
            return book['isbn']

'''
Returns True and -1,-1 if a specific bookstore location is open at a given time.
False and that locations opening and closing ints otherwise.
Time is stored as a 24 hour int for simplicity.
time passed as a parameter should follow this.
location should be passed as the exact location name as per the locations.json datatset.
'''
def isLocOpenAtTime(location: str, time: int):
    for loc in getLocationsJSON['locations']:
        if loc['name'] == location:
            if loc['open'] <= time and loc['close'] > time:
                # location found and open.
                return True, -1, -1
            else:
                # Location found and not open.
                return False, loc['open'], loc['close']
    # Location not found
    return False, -1, -1
    
'''
Returns True if a specific bookstore location is open on a given date.
False otherwise.
Input is the unix epoch timestamp for that date as an integer. 
'''
def isLocOpenOnDate(loc: str, date: int) -> Tuple[bool,str]:
    dt = datetime.datetime.fromtimestamp(date)
    weekday = dt.weekday()
    openings = None
    for loc in getLocationsJSON()['locations']:
        if loc['name'] == loc:
            openings = loc['days'] 
            break
    if openings and openings[weekday] == '0':
        return False, openings
    return True, ""

'''
Places the order as per data collected and slots filled in order().
Will decrement the stock count in stock.json.
Will update orders.json with these details.
'''
def storeOrder(title, isbn, quantity, pickup, address, date, time, cost, name):
    orders = getOrdersJSON()
    order = {
        "title": title,
        "ISBN": isbn,
        "quantity": quantity,
        "pickup": pickup,
        "address": address,
        "date": date,
        "time": time,
        "cost": cost,
        "name": name
    }
    if orders:
        orders['orders'].append(order)
    else:
        orders = {
                "orders": [order]
        }
    ordersStr = json.dumps(orders, indent=4)
    with open('orders.json', 'w') as f:
        f.write(ordersStr)
    stock = getStockJSON()
    for book in stock['stock']:
        if book['name'].lower().strip() == title:
            book['count'] -= quantity
    stockStr = json.dumps(stock, indent=4)
    with open('stock.json', 'w') as f:
        f.write(stockStr)
