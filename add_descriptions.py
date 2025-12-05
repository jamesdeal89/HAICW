import json

descriptions = {
    "Sapiens": "A sweeping history of humankind from the evolution of archaic human species to the present, exploring how Homo sapiens came to dominate the planet. Harari examines major revolutions in human history including the cognitive, agricultural, and scientific revolutions. The book challenges readers to reconsider their assumptions about progress, happiness, and what it means to be human.",
    
    "The Lord of the Rings": "Frodo Baggins inherits a powerful ring that must be destroyed in the fires of Mount Doom to prevent the dark lord Sauron from conquering Middle-earth. Accompanied by a fellowship of diverse companions, Frodo undertakes a perilous journey across a richly imagined world of elves, dwarves, wizards, and men. An epic tale of good versus evil, friendship, and sacrifice.",
    
    "Foundation": "Mathematician Hari Seldon uses psychohistory to predict the fall of the Galactic Empire and establishes a Foundation to preserve knowledge and shorten the coming dark age. The novel follows the Foundation through multiple crises as it grows from a small colony of scientists into a political power. A classic of science fiction exploring themes of history, politics, and the power of knowledge.",
    
    "The Handmaid's Tale": "In the totalitarian Republic of Gilead, women have been stripped of all rights and Offred serves as a Handmaid, forced to bear children for the ruling class. Through her eyes, we see a theocratic regime built on oppression, environmental catastrophe, and declining fertility rates. A haunting vision of religious fundamentalism and the subjugation of women.",
    
    "Brave New World": "In a technologically advanced future, humans are genetically engineered and conditioned from birth to fit predetermined social castes, kept docile through pleasure and drugs. Bernard Marx and John the Savage challenge this seemingly perfect society, questioning the price of stability and happiness. A prescient satire exploring the dangers of technology, consumerism, and the loss of individuality.",
    
    "The Name of the Wind": "Kvothe, now an innkeeper, recounts his transformation from a gifted child in a traveling troupe to a legendary figure known for magic, music, and adventure. His tale involves his time at a prestigious magic university, his quest to learn the truth about the mysterious Chandrian who killed his family, and his pursuit of the beautiful Denna. A lyrical fantasy that blends music, magic, and mystery.",
    
    "American Gods": "Ex-convict Shadow Moon is recruited by the mysterious Mr. Wednesday for a cross-country journey that reveals ancient gods living forgotten in modern America. These old gods face a coming war with new deities of technology, media, and celebrity. A darkly imaginative blend of mythology, Americana, and fantasy exploring belief, identity, and cultural change.",
    
    "Educated": "Tara Westover grows up in rural Idaho with survivalist parents who keep her out of school, working in her father's junkyard. At seventeen, she enters a classroom for the first time and eventually earns a PhD from Cambridge University. A remarkable story of the transformative power of education and the painful cost of breaking away from family.",
    
    "The Silent Patient": "Alicia Berenson, a famous painter, shoots her husband and then never speaks another word. Psychotherapist Theo Faber becomes obsessed with uncovering her motive and getting her to talk. A psychological thriller with a shocking twist about obsession, trauma, and the unreliability of memory.",
    
    "It Ends with Us": "Lily Bloom opens a flower shop in Boston and falls for neurosurgeon Ryle Kincaid, but their relationship becomes complicated when her first love Atlas reappears. As Lily navigates difficult choices, she must confront the cycle of domestic violence she witnessed in her childhood. A powerful and emotional story about love, strength, and breaking destructive patterns.",
    
    "The Seven Husbands of Evelyn Hugo": "Aging Hollywood icon Evelyn Hugo finally tells the story of her glamorous and scandalous life to unknown magazine reporter Monique Grant. Through seven marriages and decades of fame, Evelyn reveals the truth about her life, loves, and the price of ambition. A captivating tale of Old Hollywood, forbidden love, and the secrets we keep.",
    
    "The Book Thief": "Narrated by Death, the story follows Liesel Meminger, a foster girl living in Nazi Germany who steals books and shares them with others, including the Jewish man hiding in her basement. Set during World War II, Liesel finds solace in stories while witnessing the horrors of war and the Holocaust. A moving tale about the power of words, love, and humanity in the darkest times.",
    
    "Atomic Habits": "A practical guide to building good habits and breaking bad ones through tiny changes that deliver remarkable results. Clear explains the science behind habit formation and provides strategies for making habits obvious, attractive, easy, and satisfying. The book demonstrates how small improvements compound over time to produce extraordinary outcomes.",
    
    "Project Hail Mary": "Ryland Grace wakes up alone on a spaceship with no memory of how he got there, discovering he's on a desperate mission to save Earth from an extinction-level threat. Using science and ingenuity, he must solve an impossible problem, aided by an unexpected alien ally. A thrilling hard science fiction adventure about survival, problem-solving, and interspecies cooperation.",
    
    "The Martian": "Astronaut Mark Watney is stranded alone on Mars after his crew evacuates during a storm, believing him dead. Using his engineering skills, humor, and determination, he must find ways to survive while NASA works to bring him home. A gripping tale of human resilience and problem-solving in the face of impossible odds.",
    
    "Snow Crash": "In a near-future America where corporations and franchises have replaced the government, hacker and pizza delivery driver Hiro Protagonist investigates a new computer virus that also affects users in the real world. The story blends virtual reality, ancient Sumerian mythology, linguistics, and anarcho-capitalism. A wild cyberpunk adventure that predicted aspects of the modern internet and virtual worlds.",
    
    "Do Androids Dream of Electric Sheep": "In post-apocalyptic San Francisco, bounty hunter Rick Deckard must track down and retire six escaped androids while questioning what separates humans from artificial beings. Most animals are extinct, and owning real animals has become a status symbol, while empathy is the key trait distinguishing humans from androids. A philosophical exploration of consciousness, empathy, and what it means to be human.",
    
    "The Hunger Games": "In dystopian Panem, sixteen-year-old Katniss Everdeen volunteers to take her sister's place in the Hunger Games, a televised fight to the death between children from twelve districts. She must use her survival skills and navigate deadly political games to stay alive and protect those she loves. A gripping story about survival, sacrifice, and rebellion against oppression.",
    
    "A Game of Thrones": "Noble families vie for control of the Iron Throne in the Seven Kingdoms of Westeros, where summers span decades and winters can last a lifetime. Political intrigue, betrayal, and warfare unfold while an ancient evil awakens in the far north. An epic fantasy featuring complex characters, moral ambiguity, and unexpected plot twists.",
    
    "Harry Potter and the Philosopher's Stone": "Orphaned Harry Potter discovers on his eleventh birthday that he's a wizard and attends Hogwarts School of Witchcraft and Wizardry. He makes friends, learns magic, and uncovers the truth about his parents' death and his connection to the dark wizard Voldemort. The beginning of a beloved series about friendship, bravery, and the battle between good and evil.",
    
    "The Way of Kings": "On the storm-ravaged world of Roshar, multiple characters navigate war, politics, and magic while an ancient evil threatens to return. Kaladin, a slave soldier, must protect those he hates while Shallan seeks to save her family through theft and deception. An epic fantasy with innovative magic systems, complex worldbuilding, and themes of honor and redemption.",
    
    "Mistborn: The Final Empire": "In a world where ash falls from the sky and the immortal Lord Ruler has reigned for a thousand years, a street thief named Vin discovers she has magical abilities. She joins a crew of rebels planning to overthrow the empire using Allomancy, a magic system based on ingesting and burning metals. A heist story combined with epic fantasy featuring innovative magic and political intrigue.",
    
    "The Girl with the Dragon Tattoo": "Journalist Mikael Blomkvist is hired to investigate the forty-year-old disappearance of a wealthy industrialist's niece, aided by brilliant but troubled hacker Lisbeth Salander. Their investigation uncovers dark family secrets and a series of brutal murders. A gripping Swedish thriller about corruption, violence against women, and the pursuit of justice.",
    
    "Gone Girl": "When Amy Dunne disappears on her fifth wedding anniversary, suspicion falls on her husband Nick as evidence mounts against him. Told in alternating perspectives, the story reveals the dark truth about their marriage and Amy's elaborate revenge plot. A twisted psychological thriller about manipulation, media sensationalism, and the facades of marriage.",
    
    "The Da Vinci Code": "Harvard symbologist Robert Langdon is called to the Louvre when a curator is murdered with cryptic symbols left behind. Teaming with cryptologist Sophie Neveu, Langdon follows clues through art, architecture, and secret societies, uncovering a conspiracy involving the Catholic Church and the Holy Grail. A fast-paced thriller blending history, religion, and mystery.",
    
    "Big Little Lies": "Three women in a seaside Australian town become entangled in secrets and lies that culminate in a murder at a school fundraiser. Beneath the surface of perfect suburban lives lie domestic violence, sexual assault, and fierce friendship. A darkly comic mystery about motherhood, marriage, and the dangerous secrets women keep.",
    
    "All the Light We Cannot See": "Blind French girl Marie-Laure and German orphan Werner's lives intersect in occupied France during World War II. Marie-Laure flees Paris with her father and a valuable diamond, while Werner's gift for repairing radios leads him to a Nazi training school. A beautifully written story about survival, humanity, and the unseen connections that bind us.",
    
    "The Nightingale": "Two French sisters navigate occupied France during World War II in vastly different ways: Vianne struggles to survive while protecting a Jewish child, and Isabelle joins the Resistance. Their choices test their courage, loyalty, and the bonds between them. A powerful story about women's resilience and sacrifice during wartime.",
    
    "The Pillars of the Earth": "In 12th-century England, Prior Philip dreams of building a cathedral while master builder Tom and his family struggle to survive. Their lives intertwine with nobles, monks, and outlaws as civil war and religious power struggles shape medieval England. An epic historical saga spanning decades, filled with ambition, betrayal, and the human cost of grand architectural achievement.",
    
    "Where the Crawdads Sing": "Kya Clark, the Marsh Girl, grows up isolated in the North Carolina marshlands, abandoned by her family and shunned by the nearby town. When a local man is found dead, Kya becomes the prime suspect in a murder investigation. A haunting coming-of-age story blending romance, mystery, and stunning nature writing.",
    
    "Normal People": "Connell and Marianne circle each other from their school days in small-town Ireland through university in Dublin, their relationship shifting between friendship, love, and painful separation. Both struggle with class differences, mental health, and the complexity of human connection. A nuanced exploration of intimacy, miscommunication, and the lasting impact of first love.",
    
    "Little Fires Everywhere": "In suburban Ohio, the arrival of artist Mia Warren and her daughter Pearl upends the carefully ordered life of the Richardson family. Tensions escalate over a custody battle and buried secrets, ultimately leading to a devastating fire. A compelling examination of motherhood, privilege, identity, and the danger of perfection.",
    
    "Becoming": "Former First Lady Michelle Obama chronicles her journey from childhood on Chicago's South Side through her time in the White House. She reflects on her experiences as a mother, wife, lawyer, and public figure, sharing personal insights about finding her voice and staying true to herself. An inspiring memoir about identity, purpose, and navigating public life.",
    
    "The Glass Castle": "Jeannette Walls recounts her unconventional childhood with brilliant but dysfunctional parents who chose a nomadic, impoverished lifestyle. Despite hunger, neglect, and instability, the Walls children forge bonds and eventually escape to build successful lives. A remarkable story of resilience, forgiveness, and the complexity of family love.",
    
    "Born a Crime": "Comedian Trevor Noah tells stories of growing up in South Africa during and after apartheid, born to a black mother and white father when their relationship was illegal. His mother's fierce determination and unconventional parenting shape his survival in a violent, unjust society. A funny, heartbreaking memoir about identity, poverty, and the triumph of the human spirit.",
    
    "Thinking, Fast and Slow": "Nobel laureate Daniel Kahneman explores the two systems that drive human thinking: fast, intuitive thinking and slow, deliberative thinking. He examines cognitive biases, heuristics, and how these mental shortcuts lead to both effective decisions and systematic errors. A groundbreaking exploration of judgment, decision-making, and behavioral economics.",
    
    "A Brief History of Time": "Renowned physicist Stephen Hawking explains complex concepts about the universe including the Big Bang, black holes, and the nature of time in accessible language. He explores fundamental questions about the origin and fate of the universe and humanity's place within it. A landmark work making cosmology understandable to general readers.",
    
    "The Immortal Life of Henrietta Lacks": "The story of Henrietta Lacks, whose cancer cells were taken without her knowledge in 1951 and became one of the most important tools in medicine, leading to countless scientific breakthroughs. Skloot explores the ethical implications while following Henrietta's family's struggle with poverty and the legacy of those cells. A powerful intersection of race, ethics, science, and family.",
    
    "The Power of Now": "Eckhart Tolle presents a guide to spiritual enlightenment through living in the present moment and freeing yourself from ego-based thinking. He argues that psychological time and the mind-made self cause suffering and that true peace comes from presence. A transformative exploration of consciousness, mindfulness, and inner peace.",
    
    "The Subtle Art of Not Giving a F*ck": "Mark Manson argues that life's struggle is what gives it meaning and that we should choose carefully what we care about instead of trying to be positive about everything. Using profanity-laced straight talk, he challenges conventional self-help wisdom. A counterintuitive approach to living a good life by embracing limitations and choosing values wisely.",
    
    "How to Win Friends and Influence People": "Dale Carnegie's classic guide offers practical advice for improving interpersonal skills, handling people, winning them over, and changing their behavior without arousing resentment. Based on his years teaching business courses, the book emphasizes the importance of genuine interest in others and understanding their perspectives. Timeless principles for building relationships and leadership.",
    
    "Jane Eyre": "Orphaned Jane Eyre endures a harsh childhood and finds work as a governess at Thornfield Hall, where she falls in love with her employer, Mr. Rochester. Their romance is disrupted by the revelation of a terrible secret hidden in the attic. A groundbreaking novel featuring a strong, independent female protagonist navigating morality, social class, and love.",
    
    "Wuthering Heights": "The passionate and destructive relationship between Heathcliff and Catherine Earnshaw spans two generations on the Yorkshire moors. After Catherine marries another man, Heathcliff's obsessive revenge consumes everyone around them. A dark Gothic romance about love, cruelty, and the destructive power of obsession.",
    
    "Fahrenheit 451": "In a future where firemen burn books instead of extinguishing fires, Guy Montag begins to question his role in suppressing knowledge and free thought. After meeting a free-thinking young woman, he risks everything to preserve literature and ideas. A classic dystopian warning about censorship, conformity, and the power of books.",
    
    "The Road": "A father and son journey through a post-apocalyptic America, struggling to survive while maintaining their humanity in a world of ash and cannibals. Their bond and determination to reach the coast drive them forward through unimaginable hardship. A bleak but beautiful meditation on love, survival, and hope in the face of extinction.",
    
    "Ender's Game": "Brilliant child Ender Wiggin is recruited into Battle School to prepare for an alien invasion, where he excels at war games and military strategy. As he rises through the ranks, Ender faces isolation, moral dilemmas, and psychological manipulation. A thought-provoking tale about leadership, empathy, and the cost of turning children into weapons.",
    
    "The Hitchhiker's Guide to the Galaxy": "Arthur Dent is rescued moments before Earth's destruction by his friend Ford Prefect, who reveals he's an alien researcher for a galactic travel guide. They embark on absurd adventures across the universe involving depressed robots, hyperspace bypasses, and the answer to life, the universe, and everything. A hilarious satire of science fiction and life itself.",
    
    "Circe": "Circe, daughter of the sun god Helios, is banished to a deserted island after discovering her power of witchcraft. Over centuries, she hones her craft and encounters famous mythological figures including Odysseus, while navigating the cruelty of gods and mortals. A feminist retelling of Greek mythology exploring power, transformation, and self-discovery.",
    
    "The Song of Achilles": "Patroclus, an exiled prince, is sent to King Peleus's court where he meets the legendary Achilles, and they form a deep bond. Their relationship evolves from friendship to love as they navigate Achilles's destiny to become a great warrior and face the Trojan War. A beautiful reimagining of the Iliad focused on love, fate, and heroism.",
    
    "The Kite Runner": "Amir, a privileged boy in Kabul, betrays his loyal friend Hassan and spends decades haunted by guilt. After fleeing to America during the Soviet invasion, he returns to Taliban-ruled Afghanistan seeking redemption. A powerful story of friendship, betrayal, and the enduring bonds of the past against the backdrop of Afghanistan's tumultuous history.",
    
    "Life of Pi": "After a shipwreck, sixteen-year-old Pi Patel survives 227 days adrift in the Pacific Ocean on a lifeboat with a Bengal tiger named Richard Parker. His struggle for survival tests his faith, ingenuity, and understanding of the boundaries between humans and animals. A philosophical adventure about storytelling, belief, and the nature of truth.",
    
    "The Alchemist": "Santiago, a Spanish shepherd boy, dreams of treasure in Egypt and embarks on a journey to follow his Personal Legend. Along the way, he learns about the Soul of the World and discovers that the treasure was within him all along. An allegorical novel about following dreams, listening to your heart, and finding your purpose."
}

# Read the JSON file
with open('stock.json', 'r') as f:
    data = json.load(f)

# Add descriptions to each book
for book in data['stock']:
    if book['name'] in descriptions and 'description' not in book:
        book['description'] = descriptions[book['name']]

# Write back to file
with open('stock.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Added descriptions to all books!")
