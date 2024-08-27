from game import *
import numpy as np
import time
import pygame


def debugBoardSurface(board, blockSize: int, strokeSize: int):
    dimensions = np.shape(board)
    surface = pygame.Surface((dimensions[1] * blockSize + 2 * strokeSize, dimensions[0] * blockSize + 2 * strokeSize))

    for cols in range(dimensions[1]):
        for rows in range(dimensions[0]):
            if board[rows, cols] != 0:
                #print(Tetris.pieceColours[int(board[rows,cols])])
                #print(pygame.Color(Tetris.pieceColours[int(board[rows,cols])]))
                surface.fill(Tetris.boardColours[int(board[rows,cols])], (
                (cols * blockSize + strokeSize, rows * blockSize + strokeSize), (blockSize, blockSize)))
            else:
                surface.fill("black", (
                (cols * blockSize + strokeSize, rows * blockSize + strokeSize), (blockSize, blockSize)))
            if not strokeSize == 0:
                pygame.draw.rect(surface, (49,49,49),
                             ((cols * blockSize + strokeSize, rows * blockSize + strokeSize), (blockSize, blockSize)),
                             strokeSize)
    return surface

# quit()
# pygame stuff
pieceColours = [
            "#000000",  # black background
            "#0f9bd7",  # I
            "#e39f02",  # O
            "#2141c6",  # J
            "#fd6d0d",  # L
            "#d70f37",  # S
            "#59b101",  # Z
            "#af298a",  # T
        ]
shadowColours = [pygame.Color.correct_gamma(pygame.Color(pieceColours[i]),1.7) for i in range(1,8)]
boardColours = pieceColours + shadowColours
for item in boardColours:
    print(pygame.Color(item))
#quick debug board printer
pygame.init()
pygame.font.init()
blockSize = 25
screen = pygame.display.set_mode((1280, 720))
strokeWidth = 0
GRAVITY = pygame.event.custom_type()
clock = pygame.time.Clock()
running = True
pos = (10, 50)
game = Tetris()
print(game.board)
print(np.shape(game.playableBoard))

pygame.time.set_timer(GRAVITY, 500)
while running:
    events = pygame.event.get()
    if any(event.type == pygame.QUIT for event in events):
        running = False
    if any(event.type == pygame.KEYDOWN and event.key == pygame.K_F4 for event in events):
        game = Tetris()
    # for event in pygame.event.get():
    #     if event.type == pygame.QUIT:
    #         running = False
    #     if event.type == pygame.KEYDOWN:
    #         if event.key == pygame.K_x:
    #             game.rotate(clockwise=True)
    #         elif event.key == pygame.K_z:
    #             game.rotate(clockwise=False)
    #         elif event.key == pygame.K_DOWN:
    #             game.move(Movement.DOWN)
    #         elif event.key == pygame.K_LEFT:
    #             game.LDAS.startTimer()
    #             game.move(Movement.LEFT)
    #         elif event.key == pygame.K_RIGHT:
    #             game.RDAS.startTimer()
    #             game.move(Movement.RIGHT)
    #         elif event.key == pygame.K_UP:
    #             game.move(Movement.HARD)
    #         elif event.key == pygame.K_SPACE:
    #             game.holdPiece()
    #         elif event.key == pygame.K_F4:
    #             game = Tetris()
    #     if event.type == pygame.KEYUP:
    #         if event.key == pygame.K_LEFT:
    #             game.LDAS.resetTimer()
    #         elif event.key == pygame.K_RIGHT:
    #             game.RDAS.resetTimer()
    #     if event.type == GRAVITY:
    #         game.move(Movement.DOWN)
    game.tick(events)

    screen.fill("black")
    fpsText = pygame.font.SysFont("Arial", 30)
    screen.blit(debugBoardSurface(game.overlayPiece(), blockSize, strokeWidth), (pos[0] + 4 * blockSize, pos[1]))
    screen.blit(game.holdPieceSurface(blockSize, strokeWidth), (pos))
    screen.blit(game.incomingQueueSurface(blockSize, strokeWidth), (pos[0] + (4 * blockSize) + 50 + blockSize * 10,pos[1]))
    screen.blit(fpsText.render(str(np.round(clock.get_fps(), 1)), False, "white"), (0, 0))
    screen.blit(fpsText.render(str(game.LDAS.timeElapsed()) + "   " + str(game.RDAS.timeElapsed()), False, "white"), (0,100))
    screen.blit(fpsText.render(str(game.LDAS.getShiftCount()) + "   " + str(game.RDAS.getShiftCount()), False, "white"), (0,200))
    screen.blit(fpsText.render(str(game.LOCKTIMER.timeElapsed()), False, "white"), (0,300))
    pygame.display.flip()
    clock.tick(500)


