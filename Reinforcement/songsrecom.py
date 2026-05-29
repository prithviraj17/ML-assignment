import random
import csv

# -----------------------------
# Load Songs from CSV
# -----------------------------

songs = {}

with open("songs.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        emotion = row["emotion"].strip()
        song = row["song"].strip()

        if emotion not in songs:
            songs[emotion] = []

        songs[emotion].append(song)

# -----------------------------
# Emotion Detection
# -----------------------------

def detect_emotion(text):

    text = text.lower()

    happy_words = [
        "happy", "excited", "great",
        "awesome", "joy", "good", "fantastic"
    ]

    sad_words = [
        "sad", "depressed", "cry",
        "upset", "lonely", "stress"
    ]

    angry_words = [
        "angry", "mad", "frustrated",
        "annoyed", "furious"
    ]

    for word in happy_words:
        if word in text:
            return "happy"

    for word in sad_words:
        if word in text:
            return "sad"

    for word in angry_words:
        if word in text:
            return "angry"

    return "neutral"


# -----------------------------
# Q Learning Setup
# -----------------------------

states = list(songs.keys())

alpha = 0.1
gamma = 0.9
epsilon = 0.2

Q = {}

for state in states:
    Q[state] = {}

    for song in songs[state]:
        Q[state][song] = 0.0


# -----------------------------
# Choose Song
# -----------------------------

def choose_song(state):

    if random.random() < epsilon:
        return random.choice(songs[state])

    return max(Q[state], key=Q[state].get)


# -----------------------------
# Update Q Table
# -----------------------------

def update_q(state, song, reward):

    current_q = Q[state][song]

    max_future_q = max(Q[state].values())

    new_q = current_q + alpha * (
        reward + gamma * max_future_q - current_q
    )

    Q[state][song] = round(new_q, 2)


# -----------------------------
# User Feedback
# -----------------------------

def get_reward():

    print("\nFeedback Options")
    print("1. Like (+10)")
    print("2. Listen Completely (+5)")
    print("3. Skip (-5)")
    print("4. Dislike (-10)")

    try:
        choice = int(input("Enter choice: "))
    except:
        return 0

    reward_map = {
        1: 10,
        2: 5,
        3: -5,
        4: -10
    }

    return reward_map.get(choice, 0)


# -----------------------------
# Main Program
# -----------------------------

print("=" * 60)
print("AI Emotion-Aware Music Player")
print("Songs Loaded From CSV")
print("Type 'exit' to quit")
print("=" * 60)

while True:

    text = input("\nHow are you feeling today? : ")

    if text.lower() == "exit":
        print("\nThank you for using the Music Player!")
        break

    emotion = detect_emotion(text)

    print(f"\nDetected Emotion : {emotion}")

    song = choose_song(emotion)

    print(f"\nRecommended Song : {song}")

    reward = get_reward()

    update_q(emotion, song, reward)

    print(f"\nReward Received : {reward}")

    print("\nTop Learned Songs:")

    ranked = sorted(
        Q[emotion].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for s, score in ranked[:5]:
        print(f"{s} --> {score}")

    print("\nLearning Completed.")