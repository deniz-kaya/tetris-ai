import pygame
import numpy

np = numpy
asd = np.ndarray((2,2))
asd.fill(2)
asd[1,1] = -2
print(np.negative(asd))
q = 0 # i want it to range between 0 and 3, 0,1,2,3,0,1,2,3 etc, up and down.
up = ""
down = ""
for i in range(10):
    up = up + str((i * 5) % 4) + ", "
    down = down + str((i * 3) % 4) + ", "
print(up)
print(down)




# pygame setup
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("black")
    # fill the screen with a color to wipe away anything from last frame
    if pygame.mouse.get_pressed()[0]:
        color = pygame.Color("blue")
    else:
        color = pygame.Color("pink")

    screen.fill(color, (pygame.mouse.get_pos(),(50,50)))
    fpsText = pygame.font.SysFont("Arial", 30)
    # RENDER YOUR GAME HERE
    screen.blit(fpsText.render(str(numpy.round(clock.get_fps(), 1)), False, "white"), (0,0))
    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()
