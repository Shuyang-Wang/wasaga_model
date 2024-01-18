import random
import string

# Expanded list of easy-to-remember and visualize adjectives (80 entries)
adjectives = [
    "Happy", "Sad", "Angry", "Excited", "Calm", "Brave", "Beautiful", "Ugly", "Smart", "Dumb",
    "Tall", "Short", "Big", "Small", "Strong", "Weak", "Fast", "Slow", "Funny", "Serious",
    "Kind", "Mean", "Generous", "Greedy", "Polite", "Rude", "Shy", "Confident", "Quiet", "Loud",
    "Clean", "Dirty", "Hard", "Soft", "Heavy", "Light", "Bright", "Dark", "Cold", "Hot",
    "Sweet", "Bitter", "Sour", "Spicy", "Salty", "Fresh", "Stale", "Dry", "Wet", "Tasty",
    "Delicious", "Healthy", "Sick", "Happy", "Sad", "Beautiful", "Ugly", "Easy", "Difficult",
    "Comfortable", "Uncomfortable", "Expensive", "Cheap", "Rich", "Poor", "Tired", "Energetic",
    "Safe", "Dangerous", "Clean", "Messy", "Honest", "Dishonest", "Clever", "Stupid", "Creative",
    "Dull", "Patient", "Impatient", "Proud", "Embarrassed", "Satisfied", "Hungry", "Thirsty",
    "Brave", "Cowardly", "Curious", "Bored", "Wise", "Foolish", "Curly", "Straight", "Long",
    "Short", "Round", "Square", "Smooth", "Rough", "Thin", "Thick", "Tender", "Tough",
    "Friendly", "Hostile", "Caring", "Selfish", "Gentle", "Harsh", "Hopeful", "Hopeless", "Loyal",
    "Disloyal", "Famous", "Unknown", "Crisp", "Soggy", "Spacious", "Cramped", "Ancient", "Modern",
    "Innocent", "Guilty", "Vivid", "Faded", "Vibrant", "Dull", "Fantastic", "Awful", "Narrow",
    "Wide", "Fragile", "Sturdy", "Tiny", "Huge", "Joyful", "Miserable", "Incredible", "Ordinary",
    "Anxious", "Calm", "Responsible", "Irresponsible", "Unique", "Common", "Elegant", "Clumsy",
    "Fragrant", "Stinky", "Radiant", "Dreary", "Glorious", "Shameful", "Adorable", "Repulsive",
    "Magical", "Boring", "Remarkable", "Forgettable", "Adventurous", "Cautious", "Eager", "Reluctant",
    "Charming", "Repulsive", "Glamorous", "Rugged"
]

# You can access the elements of this list like common_adjectives[0], common_adjectives[1], and so on.


# Expanded list of easy-to-remember and visualize nouns (80 entries)
nouns = [
    "Cat", "Dog", "House", "Car", "Book", "Tree", "Chair", "Table", "Phone", "Computer",
    "Sun", "Moon", "Star", "Flower", "Bird", "Fish", "Ball", "Baby", "Child", "Parent",
    "Friend", "Food", "Water", "Air", "Money", "Time", "Road", "School", "Teacher", "Student",
    "Doctor", "Nurse", "Hospital", "Bed", "Tooth", "Hair", "Eye", "Ear", "Hand", "Foot",
    "Heart", "Brain", "Smile", "Laugh", "Song", "Dance", "Movie", "Music", "Game", "Love",
    "Peace", "War", "Day", "Night", "Week", "Month", "Year", "Country", "City", "Street",
    "Park", "Beach", "River", "Mountain", "Forest", "Ocean", "Island", "Desert", "Flower",
    "Vegetable", "Fruit", "Animal", "Insect", "Fish", "Reptile", "Bird", "Mammal", "Vehicle",
    "Bicycle", "Bus", "Train", "Plane", "Boat", "Ship", "Truck", "Building", "Bridge", "Tower",
    "Castle", "Statue", "Museum", "Restaurant", "Cafe", "Store", "Market", "Mall", "Beach",
    "Lake", "Pool", "Stadium", "Hat", "Shirt", "Pants", "Shoes", "Glasses", "Watch", "Ring", 
    "Bracelet", "Earring", "Necklace", "Hat", "Jacket", "Umbrella", "Scarf", "Gloves", "Socks", 
    "Dress", "Suit", "Tie", "Belt", "Wallet", "Key", "Lock", "Map", "Globe", "Clock", "Calendar",
    "Camera", "Phone", "Tablet", "Chair", "Couch", "Bed", "Desk", "Lamp", "Mirror", "Window", 
    "Door", "Stairs", "Ceiling", "Floor", "Wall", "Roof", "Garden", "Park", "Zoo", "Library",
    "Airport", "Train station", "Bus stop", "Harbor", "Beach", "Lake", "River", "Mountain",
    "Forest", "Cave", "Bridge", "Tunnel", "Skyscraper", "Apartment", "House", "Fence", "Gate",
    "Porch", "Balcony", "Patio", "Driveway", "Sidewalk", "Road", "Highway", "Street", "Alley"
]

def random_alphanumeric(length):
    letters_and_digits = string.ascii_uppercase + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))

# Function to generate a random name
def generate_random_name_plus():
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    alphanumeric = random_alphanumeric(2)  # For example, a 4-character string

    # Combine an adjective, a noun, and an alphanumeric string
    name = f"{alphanumeric}_{adjective}_{noun}"
    return name


# Function to generate a random name
def generate_random_name():
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    alphanumeric = random_alphanumeric(2)  # For example, a 4-character string

    # Combine an adjective, a noun, and an alphanumeric string
    name = f"{adjective}_{noun}"
    return name