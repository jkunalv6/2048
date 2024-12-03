import pygame
import random
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

pygame.init()

FPS = 60
score = 0
move_count = 0 


WIDTH, HEIGHT = 1000,800
ROWS =4
COLS = 4

RECT_HEIGHT = 800 //ROWS
RECT_WIDTH = 800 // COLS

OUTLINE_COLOR = (187,173,160)

OUTLINE_THICKNESS = 10 
BACKGROUND_COLOR = (205,192,180)
FONT_COLOR = (119,110,101)

FONT = pygame.font.SysFont("comicsans", 60, bold= True)
MOVE_VEL = 20

WINDOW = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption('2048')
score = 0 

class Tile:
    COLORS = [
        (237,229,218),
        (238,225,201),
        (243,178,122),
        (246,150,101),
        (247,124,95),
        (247,95,59),
        (237,208,115),
        (237,204,99),
        (236,202,80),
        (173, 216, 230),
        (0, 0, 255),
        (0, 0, 139),
    ]

    def __init__(self,value,row,col):
        self.value= value 
        self.row = row
        self.col = col
        self.x = col * RECT_WIDTH
        self.y = row*RECT_HEIGHT

    def get_color(self):
        color_index = int(math.log2(self.value)) -1
        color = self.COLORS[color_index]
        return color

    def draw(self,window):
        color=self.get_color()
        pygame.draw.rect(window,color,(self.x,self.y,RECT_HEIGHT,RECT_HEIGHT))

        text = FONT.render(str(self.value), 1 , FONT_COLOR)
        window.blit(text, (self.x + (RECT_WIDTH/2 - text.get_width()/2), (self.y + (RECT_HEIGHT/2 - text.get_height()/2)) ), )


    def set_pos(self, ceil=False):
        if ceil:
            self.row = math.ceil(self.y /RECT_HEIGHT)
            self.col = math.ceil(self.x/RECT_WIDTH)
        else :
            self.row = math.floor(self.y/RECT_HEIGHT)
            self.col = math.floor(self.x/RECT_WIDTH)
    def move(self, delta):
        self.x += delta[0]
        self.y += delta[1]

def draw_grid(window):
    for row in range(1, ROWS):
        y= row * RECT_HEIGHT
        pygame.draw.line(window, OUTLINE_COLOR,(0,y), (WIDTH-200,y), OUTLINE_THICKNESS)

    for col in range(1, COLS+1):
        x= col * RECT_WIDTH
        pygame.draw.line(window, OUTLINE_COLOR,(x,0), (x,HEIGHT), OUTLINE_THICKNESS)    

    pygame.draw.rect(window,OUTLINE_COLOR,(0,0,WIDTH,HEIGHT), OUTLINE_THICKNESS)

def draw_stats(window, score, move_count):
    # Background for stats area
    pygame.draw.rect(window, (187, 173, 160), (800, 0, 200, HEIGHT))
    
    # Render "Score" Text
    score_text = FONT.render("Score:", True, FONT_COLOR)
    score_value = FONT.render(str(score), True, FONT_COLOR)
    window.blit(score_text, (900 - score_text.get_width() // 2, 100))
    window.blit(score_value, (900 - score_value.get_width() // 2, 150))
    
    # Render "Moves" Text
    moves_text = FONT.render("Moves:", True, FONT_COLOR)
    moves_value = FONT.render(str(move_count), True, FONT_COLOR)
    window.blit(moves_text, (900 - moves_text.get_width() // 2, 300))
    window.blit(moves_value, (900 - moves_value.get_width() // 2, 350))


def draw(window, tiles):
    window.fill(BACKGROUND_COLOR)

    for tile in tiles.values():
        tile.draw(window)

    draw_grid(window)
    draw_stats(window,score,move_count)

    pygame.display.update()

def draw_game_over_box(window):
    overlay = pygame.Surface((WIDTH, HEIGHT),pygame.SRCALPHA)
    overlay.fill((0,0,0,150)) 
    window.blit(overlay,(0,0))

    pygame.draw.rect(window, (255,69,0), (300,250,400,300))
    pygame.draw.rect(window,(255,255,255),(300,250,400,300), 5)

    game_over_text = FONT.render("GAME OVER", True, (255,255,255))
    window.blit(game_over_text, ((WIDTH//2 - game_over_text.get_width()//2 ), 300) )
    pygame.display.update()

def get_random_pos(tiles):
    row = None
    col = None
    while True :
        row = random.randrange(0,ROWS)
        col = random.randrange(0,COLS)

        if f"{row}{col}" not in tiles:
            break
    return row, col

def move_tiles(window,tiles,clock,direction):
    updated = True
    blocks = set()
    global score
    if direction == "left" :
        sort_func  = lambda x: x.col
        reverse = False         
        delta = (-MOVE_VEL,0)
        boundary_check = lambda tile: tile.col == 0
        get_next_tile =lambda tile: tiles.get(f"{tile.row}{tile.col -1}")
        merge_check = lambda tile, next_tile : tile.x > next_tile.x + MOVE_VEL 
        move_check = lambda tile, next_tile : tile.x > next_tile.x + RECT_WIDTH + MOVE_VEL
        ceil = True 


    elif direction == "right":
        sort_func  = lambda x: x.col
        reverse = True         
        delta = (MOVE_VEL,0)
        boundary_check = lambda tile: tile.col == COLS -1
        get_next_tile =lambda tile: tiles.get(f"{tile.row}{tile.col +1}")
        merge_check = lambda tile, next_tile : tile.x < next_tile.x - MOVE_VEL 
        move_check = lambda tile, next_tile : next_tile.x > tile.x + RECT_WIDTH + MOVE_VEL
        ceil = False 
    elif direction == "up":
        sort_func  = lambda x: x.row
        reverse = False         
        delta = (0, -MOVE_VEL)
        boundary_check = lambda tile: tile.row == 0
        get_next_tile =lambda tile: tiles.get(f"{tile.row-1}{tile.col}")
        merge_check = lambda tile, next_tile : tile.y > next_tile.y + MOVE_VEL 
        move_check = lambda tile, next_tile : tile.y > next_tile.y + RECT_WIDTH + MOVE_VEL
        ceil = True 
    elif direction == "down":
        sort_func  = lambda x: x.row
        reverse = True         
        delta = (0, MOVE_VEL)
        boundary_check = lambda tile: tile.row == ROWS -1
        get_next_tile =lambda tile: tiles.get(f"{tile.row+1}{tile.col}")
        merge_check = lambda tile, next_tile : tile.y < next_tile.y - MOVE_VEL 
        move_check = lambda tile, next_tile : next_tile.y > tile.y + RECT_WIDTH + MOVE_VEL
        ceil = False


    while updated: 
        clock.tick(FPS)
        updated = False 
        sorted_tiles = sorted(tiles.values(), key=sort_func,reverse=reverse)

        for i, tile in enumerate(sorted_tiles):
            if boundary_check(tile):
                continue 
            next_tile = get_next_tile(tile)
            if not next_tile:
                tile.move(delta)
            elif (
                tile.value == next_tile.value and tile not in blocks and next_tile not in blocks
            ):
                if merge_check(tile,next_tile):
                    tile.move(delta)
                else:
                    next_tile.value *= 2
                    score += next_tile.value
                    sorted_tiles.pop(i)
                    blocks.add(next_tile)
            elif move_check(tile,next_tile):
                tile.move(delta)
            else :
                continue 

            tile.set_pos(ceil)
    
            updated = True 
        update_tiles(window,tiles, sorted_tiles)
    return end_move(tiles)

def is_game_over(tiles):
    if len(tiles) == 16 :
        for row in range(ROWS):
            for col in range(COLS):
                current_tile = tiles.get(f"{row}{col}")

                neighbors = [ tiles.get(f"{row}{col+1}"), tiles.get(f"{row+1}{col}")]

                for n in neighbors :
                    if n and n.value == current_tile.value :
                        return False 
        return True 
    else :
        return False 
        

def end_move(tiles):
    if len(tiles) == 16:
        return "lost"
    row, col = get_random_pos(tiles)
    new_value = random.choices([2,4],weights=[90,10], k=1)[0]
    tiles[f"{row}{col}"] = Tile(new_value, row, col) 
    return "continue"

def update_tiles(window, tiles, sorted_tiles):
    tiles.clear()
    for tile in sorted_tiles:
        tiles[f"{tile.row}{tile.col}"] = tile
    
    draw(window,tiles) 

def generate_tiles():
    tiles = {}
    for _ in range(2):
        row,col = get_random_pos(tiles)
        tiles[f"{row}{col}"] = Tile(2,row,col)
    return tiles

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128,128)
        self.fc4 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


    # Python code to find key with Maximum value in Dictionary
 
# Dictionary Initialization


# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Encode the board state for the DQN
def encode_board(tiles):
    board = np.zeros((ROWS, COLS))
    for key, tile in tiles.items():
        row, col = int(key[0]), int(key[1])
        board[row, col] = math.log2(tile.value) if tile.value > 0 else 0
    return board.flatten()

def select_action(state, model, epsilon):
    if np.random.random() < epsilon:
        return random.randint(0, 3)  # Random action
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    q_values = model(state_tensor)
    return torch.argmax(q_values).item()


# Initialize DQN components
input_size = ROWS * COLS  # Flattened board size
output_size = 4  # Four actions: up, down, left, right

# Detect GPU and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

policy_net = DQN(input_size, output_size).to(device)
target_net = DQN(input_size, output_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
memory = ReplayMemory(10000)

# Training hyperparameters
gamma = 0.99
epsilon = .5
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 64

def optimize_model():
    if len(memory) < batch_size:
        return

    # Sample a batch of experiences
    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))
    states, actions, rewards, next_states, dones = batch

    # Move data to the GPU (or CPU, depending on the device)
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Convert dones to boolean for indexing
    dones = dones.bool()

    # Compute current Q-values
    current_q = policy_net(states).gather(1, actions).squeeze(1)

    # Compute next Q-values, with zero for terminal states
    max_next_q = target_net(next_states).max(1)[0]
    max_next_q[dones] = 0.0  # Apply mask for terminal states

    # Compute target Q-values
    target_q = rewards + gamma * max_next_q

    # Compute the loss
    loss = nn.MSELoss()(current_q, target_q)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



# Main game loop with DQN training
TRAINING_MODE = True  # Set True for training without display


def main(window):
    clock = pygame.time.Clock()
    global epsilon
    
    for episode in range(1000):  # Number of training episodes
        print(f"Starting Episode {episode + 1}...")
        
        # Reset score and game board
        global score
        score = 0
        tiles = generate_tiles()  # Initialize the game state
        done = False
        state = encode_board(tiles)

        while not done:
            

            # Select an action using the policy (no need to calculate gradients here)
            with torch.no_grad():  # Skip gradient computation for action selection
                action = select_action(state, policy_net, epsilon)

            # Execute action
            directions = ["up", "down", "left", "right"]
            direction = directions[action]
            valid = move_tiles(window, tiles, clock, direction)
            next_state = encode_board(tiles)
            reward = 0 

            for x in range (0,6):
                reward += state[x]*25*(7-x)
            


            # Assign rewards
            if not valid:
                reward -= 50  # Penalty for invalid moves
            else:
                reward += score*0.8 # Reward based on current score

            done = is_game_over(tiles)
            if done:
                reward -= 100  # Penalty for losing

            # Store experience
            memory.add((state, action, reward, next_state, done))

            # Update the state for the next step
            state = next_state

            # Optimize the model
            optimize_model()

        # Decay epsilon (exploration rate)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Sync the target network every 10 episodes
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Print feedback after every 50 episodes for validation
        
        print(f"--- Completed {episode + 1} episodes ---")
        print(f"Epsilon: {epsilon:.4f}")
        print(f"Score after episode {episode + 1}: {score}\n")
        print(f"{reward}")

    print("Training complete.")
    # Save the trained model
    torch.save(policy_net.state_dict(), "dqn1_2048.pth")
    torch.save(policy_net, "dqn_2048_model1_entire.pth")
    torch.save(target_net.state_dict(), "dqn1_2048_target_net.pth")

    print("Model saved to dqn_2048.pth")


main(WINDOW)
