# The tetris game
from enum import IntEnum
import time
import torch
import numpy as np
import random as rnd

import pygame

## required:
# TODO LOCK DELAY
# TODO SOFT DROP
# TODO SUPPORT FOR S SPIN - CHANGE ORDER
settings = 117, 10, 10 # sdr: lower limit: 1 cell per 20 frames therefore 1 ---- 1-20 frames, 20-40 : -20 per frame
class TimeKeeper:
    def __init__(self):
        self.start = 0
    def startTimer(self):
        self.start = time.time_ns()
    def resetTimer(self):
        self.start = 0
    def timeElapsed(self): # ms
        if self.start == 0:
            return 0
        else:
            return (time.time_ns() - self.start) // 1000000
class TimeLocker(TimeKeeper):
    def __init__(self, lockTime: int):
        super().__init__()
        self.lockTime = lockTime
    def shoudLock(self):
        return self.timeElapsed() >= self.lockTime
    def resetTimer(self):
        if self.timeElapsed() > 0:
            self.startTimer()
    def stopTimer(self):
        self.start = 0
class DAS(TimeKeeper):
    def __init__(self, settings: [int,int,int]):
        super().__init__()
        self.das = settings[0]
        self.arr = settings[1]
        self.limit = settings[2]
        self.movedCount = 0
    def reset(self):
        self.resetMoves()
        self.resetTimer()
    def resetMoves(self):
        self.movedCount = 0
    def getShiftCount(self):
        elapsed = self.timeElapsed()

        if elapsed - self.das > 0:
            return min(self.limit, (elapsed - self.das) // self.arr)
        else:
            return 0
class Movement(IntEnum):
    LEFT = -1
    RIGHT = 1
    DOWN = 0
    HARD = 2
class Piece:
    def __init__(self, number, pos):
        self.num = number
        self.piece = np.array(Tetris.pieces[self.num])
        self.rotationID = 0
        self.pos = pos
        self.dimensions = np.shape(self.piece)
    def moveBy(self, vector: tuple[int, int]):
        self.pos = np.add(self.pos, vector)

class Tetris:
    # 'constants'
    wallKickNormal = [
        [(0, 0), (0, -1), (-1, -1), (2, 0), (2, -1)],  # 0 -> R
        [(0, 0), (0, 1), (1, 1), (-2, 0), (-2, 1)],  # R -> 2
        [(0, 0), (0, 1), (-1, 1), (2, 0), (2, 1)],  # 2 -> L
        [(0, 0), (0, -1), (1, -1), (-2, 0), (-2, -1)]  # L -> 0
    ]
    wallKickI = [
        [(0, 0), (0, -1), (1, -1), (-2, 0), (-2, -1)],  # 0 -> R
        [(0, 0), (0, -1), (0, 2), (-2, -1), (1, 2)],  # R -> 2
        [(0, 0), (0, 2), (0, -1), (-1, 2), (2, -1)],  # 2 -> L
        [(0, 0), (0, 1), (0, -2), (2, 1), (-1, -2)]  # L -> 0
    ]
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
    boardColours = [
        (0, 0, 0, 255),
        (15, 155, 215, 255),
        (227, 159, 2, 255),
        (33, 65, 198, 255),
        (253, 109, 13, 255),
        (215, 15, 55, 255),
        (89, 177, 1, 255),
        (175, 41, 138, 255), ## down from here are shadow colours
        *[(int(r * 0.5), int(g * 0.5), int(b * 0.5), 255) for r, g, b, _ in [
            (15, 155, 215, 255),
            (227, 159, 2, 255),
            (33, 65, 198, 255),
            (253, 109, 13, 255),
            (215, 15, 55, 255),
            (89, 177, 1, 255),
            (175, 41, 138, 255),
        ]]

    ]
    AIPieces = [
        [  # I
            [1, 1, 1, 1]
        ],
        [  # O
            [2, 2],
            [2, 2]
        ],
        [  # J
            [3, 0, 0],
            [3, 3, 3]
        ],
        [  # L
            [0, 0, 4],
            [4, 4, 4]
        ],
        [  # Z
            [5, 5, 0],
            [0, 5, 5]
        ],
        [  # S
            [0, 6, 6],
            [6, 6, 0]
        ],
        [  # T
            [0, 7, 0],
            [7, 7, 7]
        ]
    ]
    pieces = [
        [  # I
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ],
        [  # O
            [2, 2],
            [2, 2]
        ],
        [  # J
            [3, 0, 0],
            [3, 3, 3],
            [0, 0, 0]
        ],
        [  # L
            [0, 0, 4],
            [4, 4, 4],
            [0, 0, 0]
        ],
        [  # Z
            [5, 5, 0],
            [0, 5, 5],
            [0, 0, 0]
        ],
        [  # S
            [0, 6, 6],
            [6, 6, 0],
            [0, 0, 0]
        ],
        [  # T
            [0, 7, 0],
            [7, 7, 7],
            [0, 0, 0]
        ]
    ]
    # y by x
    bag = []
    feed = []
    boardDimensions = (27,14)
    boardWidth = 10
    boardHeight = 20
    def visibleBoard(self):
        return np.copy(self.board[5:25, 2:12])
    def setVisibleBoard(self, board):
        self.board[5:25, 2:12] = board
    def emptyBoard(self):
        b = np.zeros((27,14))
        b[:, :2] = 1
        b[:, 12:] = 1
        b[25:, :] = 1
        return b
    def newBag(self):
        self.bag = list(range(7))
        rnd.shuffle(self.bag)

    def spawnPiece(self):
        self.piecesDropped += 1
        piece_index = self.feed.pop(0)
        self.currentPiece = Piece(piece_index, (4, 6 if piece_index == 1 else 5))
        if len(self.bag) < 1:
            self.newBag()
        self.feed.append(self.bag.pop())
    def bakePiece(self):
        pos = self.currentPiece.pos
        for y in range(self.currentPiece.dimensions[0]):
            for x in range(self.currentPiece.dimensions[1]):
                 if self.board[pos[0] + y,pos[1] + x] == 0:
                    self.board[pos[0] + y,pos[1] + x] = self.currentPiece.piece[y,x]
    def getStateInfo(self, board):
        holes = self.getHoles(board)
        bumpiness, height = self.getBumpinessAndHeight(board)
        linesCleared = self.getLinesCleared(board)
        return torch.FloatTensor([holes,bumpiness,height,linesCleared])
    # TODO test linesCleared
    def getLinesCleared(self, board):
        linesCleared = 0
        for row in range(np.shape(board)[0]):
            if board[row, :].all() != 0:
                linesCleared += 1
        return linesCleared
    def solidify(self, board, piece, pos):
        dimensions = np.shape(piece)
        for y in range(dimensions[0]):
            for x in range(dimensions[1]):
                if piece[y,x] != 0:
                    board[pos[1] + y, pos[0] + x] = piece[y,x]
        return board
    def getHoles(self, board):
        holes = 0
        for column in board.T:
            startIndex = 0
            while startIndex < self.boardHeight and column[startIndex] == 0:
                startIndex += 1
            for cell in column[startIndex:]:
                holes += 1 if cell == 0 else 0
        return holes
    def getBumpinessAndHeight(self, board):
        blocks = board != 0
        # 20 by 10:
        lineHeights = np.where(blocks.any(axis=0), 20 - np.argmax(blocks, axis=0), 0)
        totalHeight = np.sum(lineHeights)
        bumpiness = 0
        for i in range(9):
            bumpiness += np.abs(lineHeights[i] - lineHeights[i+1])
        return bumpiness, totalHeight

    # TODO its 1 am i cannot be bothered with doing this efficiently, sorry future me if you profile and this takes up 10000%
    def nextState(self, action):
        xPos, rotations = action
        id = self.currentPiece.num
        piece = np.rot90(self.AIPieces[id], rotations, (1, 0))
        dimensions = np.shape(piece)
        touchingGround = False
        board = self.visibleBoard()
        yPos = 0
        while not touchingGround:
            yPos += 1
            for y in range(dimensions[0]):
                for x in range(dimensions[1]):
                    if dimensions[0] + yPos == 20:
                        touchingGround = True
                    elif board[y + yPos, x + xPos] != 0 and piece[y, x] != 0:
                        touchingGround = True
        self.setVisibleBoard(self.solidify(board, piece, (xPos, yPos)))

        gameOver = self.isGameOver()
        linesCleared = self.getLinesCleared(board)
        score = 0
        score += 1 + linesCleared**2 * 10
        score -= self.getHoles(board)
        if gameOver:
            score -= 10 # values have been meticulously pulled out of my ass!
        else:
            self.clearRows()
            self.spawnPiece()
        self.score += score
        return score, gameOver

    def getPossibleStateValues(self, board, pieceID):
        states = {}
        rotations = 4 if pieceID in [2,3,6] else 2
        if pieceID == 1:
            rotations = 1
        for rotation in range(rotations):
            piece = np.rot90(self.AIPieces[pieceID], rotation, (1, 0))
            dimensions = np.shape(piece)
            moveRightCount = 10 - dimensions[1]
            for xPos in range(moveRightCount + 1):
                yPos = -1
                #move it down until it touches the ground
                touchingGround = False
                while not touchingGround:
                    yPos += 1
                    for y in range(dimensions[0]):
                        for x in range(dimensions[1]):
                            if dimensions[0] + yPos == 20:
                                touchingGround = True
                            elif board[y + yPos, x + xPos] != 0 and piece[y,x] != 0:
                                touchingGround = True
                # store the stateValue at the
                if not yPos == 0:
                    states[(xPos, rotation)] = self.getStateInfo(self.solidify(self.visibleBoard(), piece, (xPos,yPos)))
        return states

    def getSnapppedDownPos(self):
        downBy = 0
        touchingGround = False
        while not touchingGround:
            downBy += 1
            if not self.validPlacement(self.board, self.currentPiece, (downBy, 0)):
                touchingGround = True
                downBy -= 1

        return np.add((downBy, 0), self.currentPiece.pos)
    def move(self, direction: Movement):
        # TODO something feels wrong with this subroutine
        if direction == Movement.RIGHT or direction == Movement.LEFT:
            if self.validPlacement(self.board, self.currentPiece, (0, direction)):
                self.LOCKTIMER.resetTimer()
                self.currentPiece.moveBy((0,direction))

        elif direction == Movement.DOWN:
            self.currentPiece.moveBy((1,0))
            self.SDTIMER.movedCount += 1
            if not self.validPlacement(self.board, self.currentPiece):
                self.currentPiece.moveBy((-1,0))
                self.SDTIMER.resetMoves()
                self.SDTIMER.startTimer()
                if not self.LOCKTIMER.timeElapsed() > 0:
                    self.LOCKTIMER.startTimer()
        else:
            self.currentPiece.pos = self.getSnapppedDownPos()
            self.bakePiece()
            self.clearRows()
            self.spawnPiece()

    def clearRows(self):
        emptyrow = [1,1,0,0,0,0,0,0,0,0,0,0,1,1]
        delIndexes = list()
        for row in range(np.shape(self.board)[0] - 2):
            if self.board[row, :].all() != 0:
                delIndexes.append(row)
        self.board = np.delete(self.board, delIndexes, 0)
        for i in range(len(delIndexes)):
            self.clearedLines += 1
            self.board = np.vstack([emptyrow, self.board])


    def overlayPiece(self):
        pos = self.currentPiece.pos
        ghost = self.getSnapppedDownPos()
        thisBoard = np.copy(self.board)
        for y in range(self.currentPiece.dimensions[0]):
            for x in range(self.currentPiece.dimensions[1]):
                if thisBoard[pos[0] + y, pos[1] + x] == 0:
                    thisBoard[pos[0] + y, pos[1] + x] = self.currentPiece.piece[y, x]
        for y in range(self.currentPiece.dimensions[0]):
            for x in range(self.currentPiece.dimensions[1]):
                if thisBoard[ghost[0] + y, ghost[1] + x] == 0:
                    thisBoard[ghost[0] + y, ghost[1] + x] = 0 if self.currentPiece.piece[y, x] == 0 else self.currentPiece.piece[y,x] + 7
        return thisBoard[5:25, 2:12]

    def tick(self, events):
        #clean events
        downEvents = [event for event in events if event.type == pygame.KEYDOWN]
        upEvents = [event for event in events if event.type == pygame.KEYUP]

        # reset stats
        self.PPS = round(self.piecesDropped / ((self.startTime.timeElapsed() + 1) / 1000), 2)


        if any(self.board[5,5:9] != 0):
            self.__init__()
        # reset downwards movement
        if any(event.key == pygame.K_DOWN for event in upEvents):
            self.SDTIMER.reset()

        # reset das
        if any(event.key == pygame.K_LEFT for event in upEvents):
            self.LDAS.reset()
            if not self.RDAS.timeElapsed() == 0:
                self.RDAS.resetMoves()
                self.RDAS.startTimer()
        if any(event.key == pygame.K_RIGHT for event in upEvents):
            self.RDAS.reset()
            if not self.LDAS.timeElapsed() == 0:
                self.LDAS.resetMoves()
                self.LDAS.startTimer()
        # lock
        # TODO do the snap and lock delay thing so that it makes sense

        # soft drop
        if any(event.key == pygame.K_DOWN for event in downEvents):
            self.SDTIMER.startTimer()
        #
        if self.SDTIMER.getShiftCount() > self.SDTIMER.movedCount:
            self.move(Movement.DOWN)


        # hard drop
        if any(event.key == pygame.K_UP for event in downEvents):
            self.move(Movement.HARD)

        # hold
        if any(event.key == pygame.K_SPACE for event in downEvents):
            self.holdPiece()

        ## rotation
        if any(event.key == pygame.K_z for event in downEvents):
            self.rotate(clockwise=False)
        if any(event.key == pygame.K_x for event in downEvents):
            self.rotate(clockwise=True)
        if any(event.key == pygame.K_DOWN for event in downEvents):
            self.move(Movement.DOWN)


        if ((self.LDAS.timeElapsed() > 0 and self.RDAS.timeElapsed() == 0)
                or (self.LDAS.timeElapsed() < self.RDAS.timeElapsed())):
            if self.LDAS.getShiftCount() > self.LDAS.movedCount:
                self.move(Movement.LEFT)
                self.LDAS.movedCount += 1
            elif self.LDAS.getShiftCount() > 0 and self.LDAS.getShiftCount() == self.LDAS.movedCount:
                self.move(Movement.LEFT)
        if ((self.RDAS.timeElapsed() > 0 and self.LDAS.timeElapsed() == 0)
                or (self.RDAS.timeElapsed() < self.LDAS.timeElapsed())):
            if self.RDAS.getShiftCount() > self.RDAS.movedCount:
                self.move(Movement.RIGHT)
                self.RDAS.movedCount += 1
            elif self.RDAS.getShiftCount() > 0 and self.RDAS.getShiftCount() == self.RDAS.movedCount:
                self.move(Movement.RIGHT)
        # movement
        leftMove = any(event.key == pygame.K_LEFT for event in downEvents)
        rightMove = any(event.key == pygame.K_RIGHT for event in downEvents)
        if not (leftMove and rightMove):
            if leftMove:
                self.move(Movement.LEFT)
                self.LDAS.startTimer()
                if not self.RDAS.timeElapsed() == 0:
                    self.RDAS.resetMoves()
                    self.RDAS.startTimer()
            elif rightMove:
                self.move(Movement.RIGHT)
                self.RDAS.startTimer()
                if not self.LDAS.timeElapsed() == 0:
                    self.LDAS.resetMoves()
                    self.LDAS.startTimer()

        # should i lock lmao?
        if self.LOCKTIMER.shoudLock():
            if np.array_equal(self.getSnapppedDownPos(), self.currentPiece.pos):
                self.bakePiece()
                self.clearRows()
                self.LOCKTIMER.stopTimer()
                self.spawnPiece()
            else:
                self.LOCKTIMER.stopTimer()
        # das:
        # snap if touching ground - terminating
        # do hard drop if there is any - terminating
        # do rotation if there is any
        # do soft drop if there is any
        # do DAS/movement if there is any
        # terminate
    def holdPieceSurface(self, blockSize: int, strokeSize: int):
        import pygame as pg
        surface = pg.Surface([blockSize * 4 + 2 * strokeSize, blockSize * 3 + 2 * strokeSize])
        if self.holdID == -1:
            return surface
        for y in range(np.shape(self.pieces[self.holdID])[1]):
            for x in range(np.shape(self.pieces[self.holdID])[0]):
                if self.pieces[self.holdID][x][y] != 0:
                    surface.fill(pg.Color(Tetris.boardColours[self.holdID + 1]), (
                        (y * blockSize + strokeSize, (x) * blockSize + strokeSize), (blockSize, blockSize)))
                    if not strokeSize == 0:
                        pg.draw.rect(surface, (49, 49, 49),
                                 ((y * blockSize + strokeSize, (x) * blockSize + strokeSize),
                                  (blockSize, blockSize)),
                                 strokeSize)

        return surface
    def incomingQueueSurface(self, blockSize: int, strokeSize: int):
        import pygame as pg
        surface = pg.Surface([blockSize * 4 + 2 * strokeSize,blockSize * 15 + 2 * strokeSize])
        for i in range(0, len(self.feed)):
            for y in range(np.shape(self.pieces[self.feed[i]])[1]):
                for x in range(np.shape(self.pieces[self.feed[i]])[0]):
                    if self.pieces[self.feed[i]][x][y] != 0:
                        surface.fill(pg.Color(Tetris.boardColours[self.feed[i]+1]), (
                            (y * blockSize + strokeSize, ((3 * i) + x) * blockSize + strokeSize), (blockSize, blockSize)))
                        if not strokeSize == 0:
                            pg.draw.rect(surface, (49, 49, 49),
                                         ((y * blockSize + strokeSize, ((3 * i) + x) * blockSize + strokeSize),
                                          (blockSize, blockSize)),
                                         strokeSize)
        return surface

    def holdPiece(self):
        if self.holdID == -1:
            self.holdID = self.currentPiece.num
        else:
            temp = self.currentPiece.num
            self.feed.insert(0, self.holdID)
            self.holdID = temp

        self.spawnPiece()
    def isGameOver(self):
        board = self.visibleBoard()
        if any(board[3:7,0]) != 0:
            return True
        else:
            return False
    def __init__(self):
        self.startTime = TimeKeeper()
        self.startTime.startTimer()
        self.PPS: float
        self.piecesDropped = 0
        self.score = 0
        self.clearedLines = 0
        self.boardColours.append(pygame.Color.correct_gamma(pygame.Color(self.boardColours[i]),1.7) for i in range(1,7))

        self.board = self.emptyBoard()
        self.bag = list()
        self.feed = list()

        self.LOCKTIMER = TimeLocker(500) # ms
        self.SDTIMER = DAS((0,5, 20)) # botched solution
        self.LDAS = DAS((127,10, 9))
        self.RDAS = DAS((127,10, 9))

        self.holdID = -1
        self.newBag()
        for _ in range(6):
            self.feed.append(self.bag.pop())
        self.currentPiece: Piece
        self.spawnPiece()
        #debug stuff
        # self.playableBoard = self.board[:25, 2:12]

    def rotate(self, clockwise):
        # dont fucking touch this, it works so well
        self.LOCKTIMER.resetTimer()
        wallKickTable = self.wallKickI if self.currentPiece.num == 0 else self.wallKickNormal
        rot_num = self.currentPiece.rotationID
        if clockwise:
            self.currentPiece.piece = np.rot90(self.currentPiece.piece, 1, (1, 0))
            for i in range(6):
                if i == 5:
                    self.currentPiece.piece = np.rot90(self.currentPiece.piece, -1, (1, 0))
                elif self.validPlacement(self.board, self.currentPiece, wallKickTable[rot_num][i]):
                    self.currentPiece.moveBy(wallKickTable[rot_num][i])
                    self.currentPiece.rotationID = (self.currentPiece.rotationID + 5) % 4
                    break

        else:
            rot_num = (rot_num + 3) % 4
            self.currentPiece.piece = np.rot90(self.currentPiece.piece, -1, (1, 0))
            for i in range(6):
                # TODO fix I wall kicks for anticlockwise movement
                if i == 5:
                    self.currentPiece.piece = np.rot90(self.currentPiece.piece, 1, (1, 0))
                    break
                pos = np.negative(wallKickTable[rot_num][i])
                if self.validPlacement(self.board, self.currentPiece, pos):
                    self.currentPiece.moveBy(pos)
                    self.currentPiece.rotationID = (self.currentPiece.rotationID + 3) % 4
                    break

    ## could be re-written
    def validPlacement(self, cboard, piece: Piece, posAug: tuple[int,int] = None):
        dimensions = piece.dimensions
        actualPos = piece.pos if posAug is None else np.add(piece.pos, posAug)
        # check if
        if actualPos[0] + dimensions[0] - 1 >= 27 or actualPos[1] + dimensions[1] - 1 >= 14:
            return False

        # check any overlap between board and
        for cols in range(dimensions[1]):
            for rows in range(dimensions[0]):  # fuck the range function
                if cboard[actualPos[0] + rows][actualPos[1] + cols] != 0 and piece.piece[rows,cols] != 0:
                    return False

        # the placement passed all checks and is val`id`
        return True


    # check this: https://tetris.fandom.com/wiki/SRS
    # state 0 is initial, every other state is achieved by the state number n 90 degree rotations from the initial state

