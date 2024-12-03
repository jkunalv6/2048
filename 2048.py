import pygame 
import math 
import random 
import sys

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



def main(window):
    clock = pygame.time.Clock()
    run  = True
    global move_count
    tiles = generate_tiles()

    game_over = False

    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            pygame.event.pump()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    move_count += 1
                    move_tiles(window, tiles, clock,"left")
                if event.key == pygame.K_RIGHT:
                    move_count += 1
                    move_tiles(window, tiles, clock,"right")
                if event.key == pygame.K_UP:
                    move_count += 1
                    move_tiles(window, tiles, clock,"up")
                if event.key == pygame.K_DOWN:
                    move_count += 1
                    move_tiles(window, tiles, clock,"down")
                if is_game_over(tiles):
                    game_over = True 
        
        draw(window, tiles)

        if game_over: 
            draw_game_over_box(window)

        


                
    

main(WINDOW)