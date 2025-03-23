import pygame
import random

# Constants
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 700
CARD_SIZE = 150
GRID_ROWS = 4
GRID_COLS = 4
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND_COLOR = (220, 220, 220)
SPACE = 10  # Space between cards
FLIP_DELAY = 1000  # Delay before hiding cards

# Define the card colors
colors = {
    "red": (255, 0, 0),
    "orange": (255, 165, 0),
    "yellow": (255, 255, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}

# Grid drawn
selected_colors = ["green", "yellow", "orange", "orange", 
                   "blue", "green", "red", "red", 
                   "red", "orange", "blue", "green", 
                   "yellow", "green", "red", "orange"]

# Initialize card states: False for unvisited (not flipped), and None for unknown color
card_states = {i: {"flipped": False, "color": None} for i in range(GRID_ROWS * GRID_COLS)}
colors_found = {color: [] for color in colors.keys()}  # Initialize an empty list for each color

# Initialize the Pygame window
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Memory Puzzle Game")

# Function to detect card color
def detectColor(card_id):
    # will be replaced with opencv code
    return selected_colors[card_id]

# Function to choose a random card that has not been flipped
def chooseRandomCard(): 
    unvisited_cards = [card_id for card_id, state in card_states.items() if not state["flipped"]]
    print(f"Unvisited cards: {unvisited_cards}")  # Debug
    return random.choice(unvisited_cards) if unvisited_cards else None

def showCard(card_id):
    # Check if the card's color is already identified
    if card_states[card_id]["color"] is None:
        # If not, detect the color and update the card's state
        color = detectColor(card_id)
        print(f"Showing card {card_id} with newly detected color {color}")  # Debug
        card_states[card_id] = {"flipped": True, "color": color}

        # Add the card to the colors_found dictionary
        colors_found[color].append(card_id)
    else:
        # If the color is already identified, just mark the card as flipped
        print(f"Showing card {card_id} with already identified color {card_states[card_id]['color']}")  # Debug
        card_states[card_id]["flipped"] = True
    drawGrid()

# Function to hide card (flip it back)
def hideCard(card_id):
    print(f"Hiding card {card_id}")  # Debug
    card_states[card_id]["flipped"] = False
    drawGrid()

# Function to find a pair of cards with the same color
def findPair():
    for color, card_ids in colors_found.items():
        if len(card_ids) >= 2:
            print(f"Found pair: {card_ids[0]} and {card_ids[1]} with color {color}")  # Debug
            return card_ids[0], card_ids[1]
    print("No pairs found")  # Debug
    return None

# Function to find a matching card for a flipped card
def findMatch(card_id): 
    color = card_states[card_id]["color"]
    print(f"Looking for match for card {card_id} with color {color}")  # Debug
    return next((other_card_id for other_card_id in colors_found[color] if other_card_id != card_id), None)

# Function to draw the grid of cards
def drawGrid():
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            index = i * GRID_COLS + j
            state = card_states[index]
            color = colors[state["color"]] if state["flipped"] else BLACK
            pygame.draw.rect(screen, color, 
                             (j * (CARD_SIZE + SPACE) + SPACE, 
                              i * (CARD_SIZE + SPACE) + SPACE, 
                              CARD_SIZE, CARD_SIZE))
            pygame.draw.rect(screen, BLACK, 
                             (j * (CARD_SIZE + SPACE) + SPACE, 
                              i * (CARD_SIZE + SPACE) + SPACE, 
                              CARD_SIZE, CARD_SIZE), 3)
    pygame.display.flip()

# Draw the grid and update the display
screen.fill(BACKGROUND_COLOR)
drawGrid()

# Game loop
running = True
pairs_found = 0
current_flipped_cards = []
clock = pygame.time.Clock()

while pairs_found < (GRID_ROWS * GRID_COLS // 2) and running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    print(f"\nCurrent flipped cards: {current_flipped_cards}")  # Debug
    print(f"Pairs found: {pairs_found}")  # Debug

    pygame.time.wait(FLIP_DELAY)

    # If no cards are flipped
    if len(current_flipped_cards) == 0:
        print("No cards flipped. Checking for pairs...")  # Debug
        # Check if any known pair exists in colors_found
        pair_ids = findPair()
        if pair_ids:
            print(f"Pair found: {pair_ids}")  # Debug
            # Flip both matching cards
            showCard(pair_ids[0])
            showCard(pair_ids[1])
            pygame.time.wait(FLIP_DELAY)

            # Update states
            colors_found[card_states[pair_ids[0]]["color"]].remove(pair_ids[0])
            colors_found[card_states[pair_ids[1]]["color"]].remove(pair_ids[1])
            pairs_found += 1
            print(f"Updated pairs_found: {pairs_found}")  # Debug
            pygame.time.wait(FLIP_DELAY)
            continue  # Continue to next loop iteration

        # If no known pair, choose a random card and flip it
        card_id = chooseRandomCard()
        if card_id is not None:
            print(f"Flipping random card: {card_id}")  # Debug
            showCard(card_id)
            current_flipped_cards.append(card_id)
            pygame.time.wait(FLIP_DELAY)

    # If one card is flipped
    elif len(current_flipped_cards) == 1:
        print("One card flipped. Looking for match...")  # Debug
        # Check if its match has been flipped before
        matched_card_id = findMatch(current_flipped_cards[0])
        if matched_card_id:
            print(f"Match found: {matched_card_id}")  # Debug
            # Flip the matching card
            showCard(matched_card_id)  # Fix: changed from showCard(0) to show the actual matching card
            pygame.time.wait(FLIP_DELAY)

            # Update states
            colors_found[card_states[current_flipped_cards[0]]["color"]].remove(current_flipped_cards[0])
            colors_found[card_states[matched_card_id]["color"]].remove(matched_card_id)
            pairs_found += 1
            print(f"Updated pairs_found: {pairs_found}")  # Debug
            
            # Add delay to see both matched cards before proceeding
            pygame.time.wait(FLIP_DELAY)
            
            current_flipped_cards.clear()
            continue  # Continue to next loop iteration

        # If no match found, choose another random card and flip it
        card_id = chooseRandomCard()
        if card_id is not None:
            print(f"Flipping another random card: {card_id}")  # Debug
            showCard(card_id)
            current_flipped_cards.append(card_id)
            pygame.time.wait(FLIP_DELAY)

    # If two cards are flipped
    elif len(current_flipped_cards) == 2:
        print("Two cards flipped. Checking for match...")  # Debug
        # Check if they match
        color1 = card_states[current_flipped_cards[0]]["color"]
        color2 = card_states[current_flipped_cards[1]]["color"]

        if color1 == color2:
            print("Cards match! Keeping them flipped.")  # Debug
            # If match, keep them flipped and update states
            colors_found[color1].remove(current_flipped_cards[0])
            colors_found[color2].remove(current_flipped_cards[1])
            pairs_found += 1
            print(f"Updated pairs_found: {pairs_found}")  # Debug
            pygame.time.wait(FLIP_DELAY)
        
        else:
            print("Cards do not match. Hiding them.")  # Debug
            # If no match, hide both cards
            hideCard(current_flipped_cards[0])
            hideCard(current_flipped_cards[1])
            pygame.time.wait(FLIP_DELAY)

        # Reset current_flipped_cards for the next round
        current_flipped_cards.clear()

    clock.tick(30)  # Limit the frame rate to 30 FPS

pygame.quit()