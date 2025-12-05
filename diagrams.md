# System Architecture Diagrams

## 1. System Architecture / Component Diagram

```mermaid
flowchart TD
    UI[User Interface]
  
    MAIN[main.py<br/>Main Controller<br/>Conversation Loop]
  
    PREPROC[preprocessing.py<br/>Stemming, Vectorization<br/>TF-IDF, Inverted Index]
  
    SEARCH[search.py<br/>searchIntent, question<br/>Cosine Similarity]
  
    CONTEXT[context.py<br/>sessionContext<br/>resolveEllipsis]
  
    HANDLERS[handlers.py<br/>small, discover, identity<br/>reccomend, check, opening<br/>address, facilities, locations]
  
    ORDERS[orders.py<br/>order, detectCorrection<br/>handleInputWithIntents<br/>getPickupDate, getPickupTime]
  
    NLG[nlg.py<br/>getReferringExpression<br/>aggregateOrderDetails<br/>generateContextualError]
  
    DATA[dataAccess.py<br/>fuzzySearchTitle<br/>stockCheck, storeOrder]
  
    UTILS[utils.py<br/>confirmation, wordToInt<br/>levenshteinDistance]
  
    DATASETS[(intents.csv<br/>qa.csv<br/>stock.json<br/>locations.json<br/>orders.json)]
  
    UI --> MAIN
    MAIN --> PREPROC & CONTEXT
    PREPROC & CONTEXT --> SEARCH
    SEARCH --> MAIN
    MAIN --> HANDLERS & ORDERS
    HANDLERS & ORDERS --> NLG & DATA & UTILS
    ORDERS --> CONTEXT
    DATA --> DATASETS
    NLG --> UI
```

# 2. Intent Classification Flow

```mermaid
flowchart TD
    START([User Input]) --> ELLIPSIS[Ellipsis Resolution<br/>context.resolveEllipsis]
    ELLIPSIS --> STEM[Stemming & Tokenization<br/>PorterStemmer]
    STEM --> VECTOR[Vectorization<br/>CountVectorizer]
    VECTOR --> TFIDF[TF-IDF Weighting<br/>TfidfTransformer]
    TFIDF --> INVERTED[Sparse Search<br/>Inverted Index]
    INVERTED --> COSINE[Cosine Similarity<br/>vs Training Data]
    COSINE --> THRESHOLD{Similarity > 0.5?}
  
    THRESHOLD -->|Yes| ROUTE[Route to Intent Handler]
    THRESHOLD -->|No| CONFIRM{Similarity > 0.3?}
  
    CONFIRM -->|Yes| ASK[Ask User Confirmation<br/>You are asking about X?]
    ASK --> USERCONF{User Confirms?}
    USERCONF -->|Yes| ADDDATA[Add to Training Data<br/>intents.csv]
    ADDDATA --> ROUTE
    USERCONF -->|No| ERROR[Show Error Message<br/>Try rephrasing]
  
    CONFIRM -->|No| ERROR
    ERROR --> START
  
    ROUTE --> HANDLER{Intent Type}
    HANDLER -->|small| SMALL[small talk handler]
    HANDLER -->|discover| DISCOVER[discover handler]
    HANDLER -->|identity| IDENTITY[identity handler]
    HANDLER -->|question| QA[Q&A search]
    HANDLER -->|order| ORDER[Transaction flow]
    HANDLER -->|recommend| RECOMMEND[Genre recommendation]
    HANDLER -->|check| CHECK[Order checking]
    HANDLER -->|opening| OPENING[Store hours]
    HANDLER -->|address| ADDRESS[Store address]
    HANDLER -->|facilities| FACILITIES[Store facilities]
    HANDLER -->|locations| LOCATIONS[List locations]
    HANDLER -->|stockCheck| STOCK[Stock availability]
    HANDLER -->|thank| THANK[Gratitude response]
  
    SMALL --> END([Response to User])
    DISCOVER --> END
    IDENTITY --> END
    QA --> END
    ORDER --> END
    RECOMMEND --> END
    CHECK --> END
    OPENING --> END
    ADDRESS --> END
    FACILITIES --> END
    LOCATIONS --> END
    STOCK --> END
    THANK --> END
```

## 3. Order Transaction Flow with Correction Points

```mermaid
stateDiagram-v2
    [*] --> BookSelection: order intent detected
  
    BookSelection --> QuantityInput: Book identified
    BookSelection --> BookSelection: Fuzzy search if not found
  
    QuantityInput --> DeliveryChoice: Quantity validated
    QuantityInput --> QuantityInput: Invalid quantity
  
    note right of QuantityInput
        Correction Point 1
        detectCorrection checks
        for book or quantity changes
    end note
  
    DeliveryChoice --> PickupFlow: pickup detected
    DeliveryChoice --> DeliveryFlow: delivery detected
    DeliveryChoice --> DeliveryChoice: Unclear input
  
    note right of DeliveryChoice
        Correction Point 2
        Can correct book or quantity
        while choosing delivery type
    end note
  
    state PickupFlow {
        [*] --> LocationSelection
        LocationSelection --> DateSelection: Valid location
        LocationSelection --> LocationSelection: Invalid location
    
        note right of LocationSelection
            Correction Point 3
            Can correct book or quantity
            during location selection
        end note
    
        DateSelection --> TimeSelection: Valid date
        DateSelection --> DateSelection: Invalid or closed date
    
        note right of DateSelection
            Correction Point 4
            Can correct book, quantity, location
            during date selection
        end note
    
        TimeSelection --> OrderConfirmation: Valid time
        TimeSelection --> TimeSelection: Store closed at time
    
        note right of TimeSelection
            Correction Point 5
            Can correct book or quantity
            during time selection
        end note
    }
  
    state DeliveryFlow {
        [*] --> AddressInput
        AddressInput --> OrderConfirmation: Valid address
        AddressInput --> AddressInput: Invalid address
    
        note right of AddressInput
            Correction Point 6
            Can correct book or quantity
            during address input
        end note
    }
  
    OrderConfirmation --> FeedbackCollection: Order stored
    FeedbackCollection --> [*]: Feedback collected
```

## 4. Context Management & Ellipsis Resolution

```mermaid
graph TD
    subgraph "Session Context"
        CTX[sessionContext Dictionary]
        LAST_BOOK[lastBook]
        LAST_LOC[lastLocation]
        LAST_INTENT[lastIntent]
        LAST_GENRE[lastGenre]
        LAST_QTY[lastQuantity]
        CONV_TURN[conversationTurn]
    end
  
    INPUT[User Input] --> CHECK_ELLIPSIS{Contains Ellipsis?}
  
    CHECK_ELLIPSIS -->|Just number| NUM_CONTEXT{lastIntent == order?}
    NUM_CONTEXT -->|Yes| RESOLVE_NUM[Resolve: order lastBook N copies]
    NUM_CONTEXT -->|No| PASS_NUM[Pass through]
  
    CHECK_ELLIPSIS -->|how about/what about X| ABOUT_CONTEXT{Check lastIntent}
    ABOUT_CONTEXT -->|check| RESOLVE_CHECK[Resolve: check X availability]
    ABOUT_CONTEXT -->|recommend| RESOLVE_REC[Resolve: recommend X books]
    ABOUT_CONTEXT -->|order| RESOLVE_ORD[Resolve: order X]
  
    CHECK_ELLIPSIS -->|it/that/this| PRONOUN_CONTEXT{lastIntent type?}
    PRONOUN_CONTEXT -->|opening/address/facilities| USE_LOC[Replace with lastLocation]
    PRONOUN_CONTEXT -->|Other| USE_BOOK[Replace with lastBook]
  
    CHECK_ELLIPSIS -->|at/in/from X| LOC_PATTERN{lastIntent == check?}
    LOC_PATTERN -->|Yes| RESOLVE_LOC[Resolve: check lastBook at X]
  
    RESOLVE_NUM --> ENHANCED[Enhanced Query]
    PASS_NUM --> ENHANCED
    RESOLVE_CHECK --> ENHANCED
    RESOLVE_REC --> ENHANCED
    RESOLVE_ORD --> ENHANCED
    USE_LOC --> ENHANCED
    USE_BOOK --> ENHANCED
    RESOLVE_LOC --> ENHANCED
  
    ENHANCED --> UPDATE_CTX[updateContext after handler]
    UPDATE_CTX --> CTX
  
    CTX --> LAST_BOOK
    CTX --> LAST_LOC
    CTX --> LAST_INTENT
    CTX --> LAST_GENRE
    CTX --> LAST_QTY
    CTX --> CONV_TURN
```

## 5. NLG Techniques Implementation

```mermaid
graph TB
    INPUT[User Query] --> HANDLER[Intent Handler]
  
    HANDLER --> RE[1. Referring Expressions]
    RE --> RE1[First Mention: Dune]
    RE1 --> RE2[Subsequent: it, this book, this title]
  
    HANDLER --> AGG[2. Aggregation]
    AGG --> AGG1[Order Components: Title, Quantity, Cost]
    AGG --> AGG2[Delivery Details: Location, Date, Time]
    AGG1 --> AGG3[Combined: 3 copies of Dune totaling 38.97 GBP]
    AGG2 --> AGG3
  
    HANDLER --> ERR[3. Contextual Errors]
    ERR --> ERR1[Error Type: date_invalid]
    ERR1 --> ERR2[Context: format, same_day, past]
    ERR2 --> ERR3[Tailored Message]
  
    HANDLER --> DM[4. Discourse Markers]
    DM --> DM1[Context: clarification, confirmation]
    DM1 --> DM2[Marker: Actually, To clarify, Now]
    DM2 --> DM3[Enhanced Message]
  
    HANDLER --> SUG[5. Suggestions]
    SUG --> SUG1[Type: available_genres, locations]
    SUG1 --> SUG2[Options List: Genre array]
    SUG2 --> SUG3[Output: We have books in X, Y, Z]
  
    RE2 --> OUTPUT[Natural Response]
    AGG3 --> OUTPUT
    ERR3 --> OUTPUT
    DM3 --> OUTPUT
    SUG3 --> OUTPUT
```

## 6. Inverted Index Structure

```mermaid
graph TB
    DOCS[Training Documents] --> STEM[Stemming]
    STEM --> VECTOR[Vectorization]
    VECTOR --> TFIDF[TF-IDF Weighting]
  
    TFIDF --> BUILD[Build Inverted Index]
    BUILD --> INDEX[Inverted Index<br/>Maps terms to documents]
    BUILD --> NORMS[Document Norms<br/>Precomputed for efficiency]
  
    QUERY[Query Input] --> QSTEM[Stem and Vectorize Query]
    QSTEM --> LOOKUP[Lookup Query Terms<br/>in Inverted Index]
  
    INDEX --> LOOKUP
    LOOKUP --> SPARSE[Sparse Dot Product<br/>Only documents with overlapping terms]
    NORMS --> COSINE[Normalize by Document Norms]
    SPARSE --> COSINE
    COSINE --> RESULT[Ranked Similarity Scores]
```

## 7. Data Flow Diagram

```mermaid
flowchart LR
    subgraph "Input Layer"
        USER[User Input]
    end
  
    subgraph "Processing Layer"
        RESOLVE[Ellipsis Resolution<br/>+Context]
        PREPROC[Preprocessing<br/>Stem, Vector, TF-IDF]
        CLASSIFY[Intent Classification<br/>Cosine Similarity]
    end
  
    subgraph "Handler Layer"
        ROUTER{Intent Router}
        SMALL_H[Small Talk]
        ORDER_H[Order Handler]
        QA_H[Q&A Handler]
        INFO_H[Info Handlers]
    end
  
    subgraph "Data Access Layer"
        STOCK_DB[(stock.json<br/>Books)]
        LOC_DB[(locations.json<br/>Stores)]
        ORDER_DB[(orders.json<br/>User orders)]
        SESSION_DB[(session.json<br/>User identity)]
        INTENT_DB[(intents.csv<br/>Training examples)]
        QA_DB[(qa.csv<br/>Q&As)]
    end
  
    subgraph "Output Layer"
        NLG[Natural Language<br/>Generation]
        RESPONSE[Response to User]
    end
  
    USER --> RESOLVE
    RESOLVE --> PREPROC
    PREPROC --> CLASSIFY
  
    INTENT_DB -.Training.-> PREPROC
    QA_DB -.Training.-> PREPROC
  
    CLASSIFY --> ROUTER
  
    ROUTER --> SMALL_H
    ROUTER --> ORDER_H
    ROUTER --> QA_H
    ROUTER --> INFO_H
  
    ORDER_H <--> STOCK_DB
    ORDER_H <--> LOC_DB
    ORDER_H --> ORDER_DB
    INFO_H <--> LOC_DB
    INFO_H <--> STOCK_DB
    QA_H <--> QA_DB
    SMALL_H <--> SESSION_DB
  
    SMALL_H --> NLG
    ORDER_H --> NLG
    QA_H --> NLG
    INFO_H --> NLG
  
    NLG --> RESPONSE
    RESPONSE --> USER
```

## 8. Correction Detection System

```mermaid
flowchart TD
    INPUT[User Input During Transaction] --> DETECT[detectCorrection function]
  
    DETECT --> PATTERN1{Match Pattern:<br/>sorry/no/wait/actually/i meant}
  
    PATTERN1 -->|Yes| TYPE{Correction Type?}
  
    TYPE -->|Contains Number| EXTRACT_QTY[Extract Quantity<br/>reQuantityExtract regex]
    TYPE -->|Contains Book Title| EXTRACT_BOOK[Extract Book Title<br/>After correction keywords]
    TYPE -->|Contains Location Keywords| EXTRACT_LOC[Extract Location<br/>With context keywords]
  
    EXTRACT_QTY --> CONVERT{Number Format?}
    CONVERT -->|Digit| NUM_INT[int conversion]
    CONVERT -->|Word| WORD_INT[wordToInt function]
  
    NUM_INT --> RETURN_QTY[Return: True, quantity, newQty]
    WORD_INT --> RETURN_QTY
  
    EXTRACT_BOOK --> FUZZY[Fuzzy Title Search<br/>levenshteinDistance]
    FUZZY --> MATCH{Match Found?<br/>distance less than or equal to 5}
    MATCH -->|Yes| RETURN_BOOK[Return: True, book, newBook]
    MATCH -->|No| RETURN_FALSE1[Return: False, None, None]
  
    EXTRACT_LOC --> RETURN_LOC[Return: True, location, newLoc]
  
    PATTERN1 -->|No| RETURN_FALSE2[Return:<br/>False, None, None]
  
    RETURN_QTY --> APPLY[Apply Correction]
    RETURN_BOOK --> APPLY
    RETURN_LOC --> APPLY
    RETURN_FALSE1 --> CONTINUE[Continue Normal Flow]
    RETURN_FALSE2 --> CONTINUE
  
    APPLY --> UPDATE[Update Order State:<br/>book, quantity, price]
    UPDATE --> RECALC[Recalculate Price]
    RECALC --> REASK[Re-ask Current Question]
    REASK --> LOOP[Loop Back to Input]
```

## 9. Small Talk Handler (ELIZA-style)

```mermaid
flowchart TD
    INPUT[User Small Talk Input] --> PATTERNS[Regex Pattern Matching]
  
    PATTERNS --> P1{Pattern 1:<br/>feel/feeling + emotion}
    PATTERNS --> P2{Pattern 2:<br/>feel X when Y}
    PATTERNS --> P3{Pattern 3:<br/>when Y feel X}
    PATTERNS --> P4{Pattern 4:<br/>how are you}
    PATTERNS --> P5{Pattern 5:<br/>greeting}
  
    P1 -->|Match| EXTRACT_EMOTION[Extract emotion word]
    P2 -->|Match| EXTRACT_REASON1[Extract emotion + reason]
    P3 -->|Match| EXTRACT_REASON2[Extract reason + emotion]
    P4 -->|Match| HOW_ARE[Build 'how are you' response]
    P5 -->|Match| GREET[Build greeting response]
  
    EXTRACT_EMOTION --> BUILD_E[Build response:<br/>Tell me more about<br/>why you feel X]
  
    EXTRACT_REASON1 --> FLIP_REF1[Flip reference words:<br/>me to you, my to your, I to you]
    EXTRACT_REASON2 --> FLIP_REF2[Flip reference words:<br/>me to you, my to your, I to you]
  
    FLIP_REF1 --> BUILD_R1[Build response:<br/>Why do you feel X<br/>when Y?]
    FLIP_REF2 --> BUILD_R2[Build response:<br/>Why do you feel X<br/>when Y?]
  
    HOW_ARE --> CHECK_NAME{Name stored?}
    CHECK_NAME -->|Yes| WITH_NAME[Response with name:<br/>I am well! How about you?]
    CHECK_NAME -->|No| NO_NAME[Response without name:<br/>I am well! How about you?]
  
    GREET --> GREET_CHECK{Name stored?}
    GREET_CHECK -->|Yes| GREET_NAME[Greet with name:<br/>Hi there]
    GREET_CHECK -->|No| ASK_NAME[Ask for name:<br/>Call identity handler]
  
    P1 -->|No Match| CHECK_MORE{More patterns?}
    P2 -->|No Match| CHECK_MORE
    P3 -->|No Match| CHECK_MORE
    P4 -->|No Match| CHECK_MORE
    P5 -->|No Match| CHECK_MORE
  
    CHECK_MORE -->|All Failed| GENERIC[Generic response:<br/>Tell me more.]
  
    BUILD_E --> COMBINE[Combine all matched<br/>response parts]
    BUILD_R1 --> COMBINE
    BUILD_R2 --> COMBINE
    WITH_NAME --> COMBINE
    NO_NAME --> COMBINE
    GREET_NAME --> COMBINE
    ASK_NAME --> COMBINE
    GENERIC --> COMBINE
  
    COMBINE --> OUTPUT[Return response string]
```

## 10. Transaction Input Interception

```mermaid
flowchart TD
    START[User Input During Transaction] --> HANDLE[handleInputWithIntents]
  
    HANDLE --> CHECK_QUIT{Quit/Cancel?}
    CHECK_QUIT -->|Yes| RETURN_QUIT[Return: userInput, False]
  
    CHECK_QUIT -->|No| CHECK_TYPE{Expected Type?}
  
    CHECK_TYPE -->|quantity| QTY_CHECK{Contains numbers<br/>or number words?}
    CHECK_TYPE -->|book| BOOK_CHECK{Length > 2 and<br/>not yes/no?}
    CHECK_TYPE -->|location| LOC_CHECK{Contains list or<br/>3 words or less?}
    CHECK_TYPE -->|pickup_delivery| PD_CHECK{Contains pickup or<br/>delivery keywords?}
    CHECK_TYPE -->|date| DATE_CHECK{Contains date-related<br/>keywords?}
    CHECK_TYPE -->|time| TIME_CHECK{Contains time-related<br/>keywords?}
    CHECK_TYPE -->|general| GEN_CHECK{Is yes/no?}
  
    QTY_CHECK -->|Yes| RELEVANT[Mark as<br/>Transaction Relevant]
    BOOK_CHECK -->|Yes| RELEVANT
    LOC_CHECK -->|Yes| RELEVANT
    PD_CHECK -->|Yes| RELEVANT
    DATE_CHECK -->|Yes| RELEVANT
    TIME_CHECK -->|Yes| RELEVANT
    GEN_CHECK -->|Yes| RELEVANT
  
    QTY_CHECK -->|No| NOT_REL[Not Transaction Relevant]
    BOOK_CHECK -->|No| NOT_REL
    LOC_CHECK -->|No| NOT_REL
    PD_CHECK -->|No| NOT_REL
    DATE_CHECK -->|No| NOT_REL
    TIME_CHECK -->|No| NOT_REL
    GEN_CHECK -->|No| NOT_REL
  
    RELEVANT --> RETURN_PROCESS[Return: userInput, False<br/>Process as transaction data]
  
    NOT_REL --> SEARCH_INTENT[searchIntent on input]
    SEARCH_INTENT --> INTENT_FOUND{Intent Match?}
  
    INTENT_FOUND -->|small| HANDLE_SMALL[Handle small talk<br/>Print response]
    INTENT_FOUND -->|discover| HANDLE_DISC[Handle discover<br/>Show capabilities]
    INTENT_FOUND -->|identity| HANDLE_ID[Handle identity<br/>Manage name]
    INTENT_FOUND -->|thank| HANDLE_THANK[Handle thank<br/>Acknowledge]
    INTENT_FOUND -->|question| HANDLE_QA[Handle Q&A<br/>Answer question]
    INTENT_FOUND -->|None/other| NO_INTENT[Return: userInput, False<br/>Process normally]
  
    HANDLE_SMALL --> BACK[Print: Now, back to your order...]
    HANDLE_DISC --> BACK
    HANDLE_ID --> BACK
    HANDLE_THANK --> BACK
    HANDLE_QA --> BACK
  
    BACK --> RETURN_RETRY[Return: None, True<br/>Signal to re-ask question]
  
    RETURN_QUIT --> END([Return to Caller])
    RETURN_PROCESS --> END
    NO_INTENT --> END
    RETURN_RETRY --> END
```

## 11. Recommendation System Flow

```mermaid
flowchart TD
    START([recommend intent detected]) --> CHOICE[Display choice:<br/>1. Genre-based<br/>2. Description-based]
  
    CHOICE --> INPUT_CHOICE[User enters choice]
    INPUT_CHOICE --> CHECK_QUIT{Input = 'quit'?}
    CHECK_QUIT -->|Yes| EXIT([Exit system])
    CHECK_QUIT -->|No| CHECK_CANCEL{Input contains 'cancel'?}
    CHECK_CANCEL -->|Yes| CANCEL_MSG[Print: Recommendation cancelled]
    CANCEL_MSG --> RETURN([Return])
  
    CHECK_CANCEL -->|No| CHOICE_TYPE{Choice = '2'?}
  
    CHOICE_TYPE -->|Yes - Description| DESC_LOOP[Description-based loop]
    CHOICE_TYPE -->|No - Genre| GENRE_FLOW[Genre-based flow]
  
    DESC_LOOP --> ASK_DESC[Ask: Describe the book you're looking for]
    ASK_DESC --> INPUT_DESC[User enters description]
    INPUT_DESC --> DESC_QUIT{Input = 'quit'?}
    DESC_QUIT -->|Yes| EXIT
    DESC_QUIT -->|No| DESC_CANCEL{Input contains 'cancel'?}
    DESC_CANCEL -->|Yes| CANCEL_MSG
    DESC_CANCEL -->|No| RESOLVE_DESC[Resolve ellipsis]
  
    RESOLVE_DESC --> GET_PREF[Get preferred genre<br/>from session.json]
    GET_PREF --> HAS_PREF{Preferred genre exists?}
    HAS_PREF -->|Yes| APPEND_GENRE[Append genre to description:<br/>userDesc + preferredGenre]
    HAS_PREF -->|No| USE_DESC[Use description as-is]
  
    APPEND_GENRE --> BOOK_SEARCH[bookDescSearch:<br/>TF-IDF + Inverted Index]
    USE_DESC --> BOOK_SEARCH
  
    BOOK_SEARCH --> RESULTS{Results found?}
    RESULTS -->|No| ERROR1[Print generic error]
    ERROR1 --> ASK_DESC
  
    RESULTS -->|Yes| GET_FIRST[Get first result]
    GET_FIRST --> DISPLAY_FIRST[Display book details:<br/>title, author, genre,<br/>pages, price, stock, description]
    DISPLAY_FIRST --> UPDATE_CONTEXT1[Update context: lastBook]
  
    UPDATE_CONTEXT1 --> CONFIRM1[Ask: Does this sound like<br/>what you're looking for?]
    CONFIRM1 --> USER_CONFIRM1{User confirms?}
  
    USER_CONFIRM1 -->|Yes| ACCEPT_MSG1[Print: Great! Let me know<br/>if you'd like to order...]
    ACCEPT_MSG1 --> RETURN
  
    USER_CONFIRM1 -->|No| CHECK_SECOND{More than 1 result?}
    CHECK_SECOND -->|No| REPHRASE_MSG1[Print: Try describing differently]
    REPHRASE_MSG1 --> ASK_DESC
  
    CHECK_SECOND -->|Yes| GET_SECOND[Get second result]
    GET_SECOND --> DISPLAY_SECOND[Display second book details]
    DISPLAY_SECOND --> UPDATE_CONTEXT2[Update context: lastBook]
  
    UPDATE_CONTEXT2 --> CONFIRM2[Ask: Does this sound better?]
    CONFIRM2 --> USER_CONFIRM2{User confirms?}
  
    USER_CONFIRM2 -->|Yes| ACCEPT_MSG2[Print: Great! Let me know<br/>if you'd like to order...]
    ACCEPT_MSG2 --> RETURN
  
    USER_CONFIRM2 -->|No| REPHRASE_MSG2[Print: Try describing differently]
    REPHRASE_MSG2 --> ASK_DESC
  
    GENRE_FLOW --> CHECK_PROMPT{Prompt contains genre?}
    CHECK_PROMPT -->|Yes| MATCH_GENRE1[Match genre from prompt]
    CHECK_PROMPT -->|No| ASK_GENRE[Ask: What genre interested in?]
  
    ASK_GENRE --> SHOW_GENRES[Display available genres list]
    SHOW_GENRES --> INPUT_GENRE[User enters genre]
    INPUT_GENRE --> GENRE_QUIT{Input = 'quit'?}
    GENRE_QUIT -->|Yes| EXIT
    GENRE_QUIT -->|No| RESOLVE_GENRE[Resolve ellipsis]
    RESOLVE_GENRE --> MATCH_GENRE2[Match genre from input]
  
    MATCH_GENRE1 --> GENRE_MATCHED{Genre found?}
    MATCH_GENRE2 --> GENRE_MATCHED
  
    GENRE_MATCHED -->|No| ERROR2[Print error:<br/>Try again with available genres]
    ERROR2 --> RETURN
  
    GENRE_MATCHED -->|Yes| UPDATE_LAST_GENRE[Update context: lastGenre]
    UPDATE_LAST_GENRE --> SAVE_PREF[Save to session.json:<br/>preferredGenre]
    SAVE_PREF --> GET_BOOKS[Get books in genre]
  
    GET_BOOKS --> BOOKS_EXIST{Books available?}
    BOOKS_EXIST -->|No| ERROR3[Print: No books found in genre]
    ERROR3 --> RETURN
  
    BOOKS_EXIST -->|Yes| RANDOM_SELECT[Randomly select one book]
    RANDOM_SELECT --> DISPLAY_GENRE_BOOK[Display book details:<br/>title, author, genre,<br/>pages, price, stock]
    DISPLAY_GENRE_BOOK --> UPDATE_GENRE_CONTEXT[Update context: lastBook]
    UPDATE_GENRE_CONTEXT --> RETURN
  
    style SAVE_PREF fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    style GET_PREF fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    style APPEND_GENRE fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
```

**Key Features:**
- **Two recommendation modes**: Genre-based (simple random selection) and Description-based (TF-IDF search)
- **Personalization**: Preferred genre from genre-based selections automatically enhances description-based searches
- **Fallback system**: Offers up to 2 recommendations per description search
- **Cancellation support**: Users can cancel at choice selection or description input
- **Session persistence**: Preferred genre saved to `session.json` for future searches
