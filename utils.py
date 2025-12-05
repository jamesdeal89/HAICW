import re
import datetime

'''
Shared function for basic confirmation from the user.
Simplifies code in other handlers by providing a global abstracted function.
Returns simply True if they confirmed, False is they did not (assume False if no affirmation detected for safety.)
'''
def confirmation() -> bool:
    answer = input("Please enter your prompt (QUIT to exit): ")
    if answer.lower() == "quit":
        exit()
    affirm = re.search(r"(?i)^\s*(?:yes|yep|yeah|y|sure|ok|okay|alright|affirmative|of course|definitely|certainly|sure thing|sounds good|roger)\b", answer) 
    if affirm:
        return True
    return False

'''
Takes day, month, year of a date.
Returns the integer Unix epoch timestamp.
For simple storage and standardised data representation.
'''
def getUnixEpochTimestamp(dd: int,mm: int,yyyy: int)-> int:
    date_obj = datetime.datetime(yyyy, mm, dd)
    return int(date_obj.timestamp())

'''
Convert Unix epoch timestamp into datetime object for easy printing when user queries their order.
'''
def getDateFromUnix(timestamp: int) -> str:
    date_obj = datetime.datetime.fromtimestamp(timestamp)
    return date_obj

'''
Resolves words for numbers a user may enter into their integer form for easy processing.
'''
def wordToInt(word: str) -> int:
    wordToInt = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'eleven': 11,
        'twelve': 12,
        'thirteen': 13,
        'fourteen': 14,
        'fifteen': 15,
        'sixteen': 16,
        'seventeen': 17,
        'eighteen': 18,
        'nineteen': 19,
        'twenty': 20,
        'thirty': 30,
        'forty': 40,
        'fifty': 50,
        'sixty': 60, 
        'seventy': 70,
        'eighty': 80,
        'ninety': 90,
        'hundred': 100,
        'thousand': 1000,
        'million': 1000000
    }
    if word in wordToInt:
        return wordToInt[word]
    else:
        return -1

'''
Returns the Levenshtein distance which can be used to implement a fuzzy search.
(To be applied to matching a user's desired book title, even if slightly off, to the name in the stock.json dataset.)

Core concept of Levenshtein distance:
- Measures difference between 2 strings,
- by counting how many single character edits are needed to reach one from the other.
Operations include:
- Insertion (add char)
- Deletion (remove char)
- Substitution (replace a char with another)

Below implementation uses a recursive approach:

'm' and 'n' are the current lengths being considered for each string respectively.
Initialised to len(a) and len(b).

If m is 0, the distance is the remaining chars in the other string (as we'd need n insertions.)
If n is 0, the distance is the number of remaining chars in the other string (as we'd need m deletions.)

If last characters match (a[m-1] == b[n-1]) then no need to edit that char, recurse on prefixes.

If last characters differ, recur with all 3 potential operations to match the chars.
'''
def levenshteinDistance(a: str, b: str, m: int, n: int) -> int:
    # Using a dynamic programming approach for efficiency.
    # O(m*n) time complexity.
    # How it works is by building up a 2D array where each entry dp[i][j] represents the Levenshtein distance
    # between the first i characters of string a and the first j characters of string b.
    # this means that we can build up the solution for larger substrings based on the solutions for smaller substrings.

    # Create a 2D array to store distances between prefixes of the strings.
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    # Fill dp array 
    for i in range(1, m + 1):
        # Current character of a
        ai = a[i - 1]
        # Iterate through each character of b
        for j in range(1, n + 1):
            bj = b[j - 1]
            # Cost of substitution
            cost = 0 if ai == bj else 1
            # Compute minimum cost possible between: deletion, insertion, substitution
            dp[i][j] = min(
                # deletion
                dp[i - 1][j] + 1,      
                # insertion
                dp[i][j - 1] + 1,      
                # substitution
                dp[i - 1][j - 1] + cost  
            )
    # The bottom right of the 2D DP array contains the Levenshtein distance built up
    return dp[m][n]
