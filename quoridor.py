import numpy as np
import math
from queue import PriorityQueue
import copy
import random

# represents the board of the game to be played on
class Board:
  def __init__(self, width, height):
    self.width = width
    self.height = height
    self.grid = np.full((width * 2 - 1, height * 2 - 1), ' ')

  def copy(self):
    b = Board(self.width, self.height)
    b.grid = self.grid.copy()
    return b

  # gets the contents of the square at (row, col)
  def bget(self, row, col):
    return self.grid[int(row*2), int(col*2)]

  # sets the contents of the square at (row, col)
  def bset(self, row, col, item):
    self.grid.itemset((int(row*2), int(col*2)), item)

class Path:
  def __init__(self, curr, key, prev=None):
    self.curr = curr
    self.key = key
    self.prev = prev
    self.dist_from_start = self.dist_from_start()

  def __lt__(self, obj):
    return self.key < obj.key

  def __eq__(self, obj):
    try:
      return self.curr == obj.curr
    except:
      return False

  def __hash__(self):
    return self.curr.__hash__()

  def solved(self, whiteTurn, height):
    if whiteTurn:
      return self.curr[0] == 0
    else:
      return self.curr[0] == height - 1

  def path_to_here(self):
    if self.prev == None:
      return [self.curr]
    pth = self.prev.path_to_here()
    pth.append(self.curr)
    return pth

  def dist_from_start(self):
    if self.prev == None:
      return 0
    else:
      return self.prev.dist_from_start + 1

class Game:
  def __init__(self, width = 9, height = 9, blocks = 10):
    self.whiteTurn = True
    self.board = Board(width, height)
    self.white = (height - 1, math.floor(width/2))
    self.black = (0, math.floor(width/2))
    self.blocks = {'white': blocks, 'black': blocks}
    for i in range(len(self.board.grid)):
      for j in range(len(self.board.grid[i])):
        if i % 2 == 1 and j % 2 == 1:
          self.board.grid.itemset((i, j), 'O')
        elif (i + j) % 2 == 1:
          self.board.grid.itemset((i, j), 'O')
    self.board.bset(self.white[0], self.white[1], 'W')
    self.board.bset(self.black[0], self.black[1], 'B')

  def copy(self):
    g = Game()
    g.whiteTurn = self.whiteTurn
    g.board = self.board.copy()
    g.white = self.white
    g.black = self.black
    g.blocks = self.blocks.copy()
    return g

  def check_win(self):
    if self.white[0] == 0:
      return 'White'
    elif self.black[0] == self.board.height - 1:
      return 'Black'
    else:
      return False

  def printout(self):
    return str(self.blocks['black']) + "\n" + str(self.board.grid) + "\n" + str(self.blocks['white'])

  # moves the piece in the specified direction based on who's turn it is
  def movePiece(self, target, whiteTurn = None):
    if whiteTurn == None:
      whiteTurn == self.whiteTurn
    if whiteTurn:
      self.board.bset(self.white[0], self.white[1], ' ')
      self.board.bset(target[0], target[1], 'W')
      self.white = target
    else:
      self.board.bset(self.black[0], self.black[1], ' ')
      self.board.bset(target[0], target[1], 'B')
      self.black = target

  # places a block based on orientation ('h' or 'v') and target ([r, c])
  # returns true if block is validly places, false if not
  def placeBlock(self, target, orientation):
    toCheck = []
    if orientation == 'h':
      toCheck = [[0, -1], [0, 1]]
    elif orientation == 'v':
      toCheck = [[-1, 0], [1, 0]]
    else:
      return False
    if self.board.grid[target[0], target[1]] != 'O' and self.board.grid[target[0] + toCheck[0][0], target[1] + toCheck[0][1]] != 'O' and self.board.grid[target[0] + toCheck[1][0], target[1] + toCheck[1][1]] != 'O':
      return False
    else:
      self.board.grid.itemset((int(target[0]), int(target[1])), 'C')
      self.board.grid.itemset((int(target[0] + toCheck[0][0]), int(target[1] + toCheck[0][1])), 'C')
      self.board.grid.itemset(int(target[0] + toCheck[1][0]), int(target[1] + toCheck[1][1]), 'C')
      return True

  # generate a list valid of pawn moves for whomever's turn it is
  def genNewMoves(self, whiteTurn = None):
    if whiteTurn == None:
      whiteTurn == self.whiteTurn
    if whiteTurn:
      pos = self.white
    else:
      pos = self.black
    out = self.genMoves({(pos[0], pos[1])}, (pos[0], pos[1]))
    out.remove((pos[0], pos[1]))
    return list(out)

  # adds the possible moves for a given position to the given set of positions (set of tuples (r, c))
  def genMoves(self, s, pos, empty_path=False):
    transforms = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    for t in transforms:
      new_coord = (pos[0] + t[0], pos[1] + t[1])
      if new_coord[0] >= 0 and new_coord[0] < self.board.width and new_coord[1] >= 0 and new_coord[1] < self.board.height and self.board.bget(new_coord[0] - 0.5 * t[0], new_coord[1] - 0.5 * t[1]) != 'C':
        if self.board.bget(new_coord[0], new_coord[1]) != ' ' and not empty_path:
          s = set.union(s, self.genMoves(s, new_coord, True))
        else:
          s.add(new_coord)
    return s

  # generates a list of possible blocks for whomever's turn it is and returns in the form ([list of h blocks], [list of v blocks])
  def genBlocks(self, whiteTurn = None):
    if whiteTurn == None:
      whiteTurn = self.whiteTurn
    out = {'h': [], 'v': []}
    if (self.blocks['white' if whiteTurn else 'black'] == 0):
      return out
    else:
      for i in range(1, self.board.height * 2 - 1, 2):
        for j in range(1, self.board.width * 2 - 1, 2):
          if self.board.grid[i, j+1] == 'O' and self.board.grid[i, j-1] == 'O' and self.checkValidPath((i, j), 'h'):
            out['h'].append((i, j))
          if self.board.grid[i+1, j] == 'O' and self.board.grid[i-1, j] == 'O' and self.checkValidPath((i, j), 'v'):
            out['v'].append((i, j))
      return out

  # given an open place to put a block, this returns true iff both pieces can still make it to the end of the board
  def checkValidPath(self, target=None, orientation=None):
    test_state = self.copy()
    if target != None and orientation != None:
      test_state.placeBlock(target, orientation)

    pos = self.white
    end = 0
    w = False
    s = set()
    s = test_state.dfs(s, pos)
    for node in s:
      if node[0] == end:
        w = True
        break

    pos = self.black
    end = self.board.height - 1
    b = False
    s = set()
    s = test_state.dfs(s, pos)
    for node in s:
      if node[0] == end:
        b = True
        break

    return w and b

  # depth first search helper function for checkValidPath, returns the set of explored coordinates
  def dfs(self, s, pos):
    s.add(pos)
    adj = list(self.genMoves(set(), pos, True))
    for move in adj:
      if move not in s:
        self.dfs(s, move)
    return s

  # returns the closest path from the current location to the end
  def path(self, whiteTurn=None):
    if not self.checkValidPath():
      return None, 0

    if whiteTurn == None:
      whiteTurn = self.whiteTurn

    if whiteTurn:
      curr = Path(self.white, self.heuristic())
    else:
      curr = Path(self.black, self.heuristic())

    Q = PriorityQueue()
    C = set()

    Q.put(curr)

    while not Q.empty():
      node = Q.get()
      if node in C:
        continue
      elif node.solved(whiteTurn, self.board.height):
        return node.path_to_here(), node.dist_from_start;
      C.add(node)
      children = list(self.genMoves(set(), node.curr, True))
      for child in children:
        new_child = Path(child, None, node)
        new_child.key = self.heuristic(child) + new_child.dist_from_start
        Q.put(new_child)

    return None, 0

  # plays the game either locally or against the cpu
  def play(self, ai=False, search_depth = 1, rand = False):
    while (self.check_win() == False):
      turn = 'white' if self.whiteTurn else 'black'
      print(self.printout())
      if turn == 'black' and rand:
        # random pick
        # print('randoming...')
        # self.whiteTurn = not self.whiteTurn
        # continue
        moves_p = self.genNewMoves(whiteTurn = False)
        states = []
        for move in moves_p:
          b_copy = self.copy()
          b_copy.movePiece(move, 0)
          states.append(b_copy)
        moves_b = self.genBlocks(turn)
        for block in moves_b['h']:
          b_copy = self.copy()
          b_copy.placeBlock(block, 'h')
          b_copy.blocks['white'] -= 1
          states.append(b_copy)
        for block in moves_b['v']:
          b_copy = self.copy()
          b_copy.placeBlock(block, 'v')
          b_copy.blocks['white'] -= 1
          states.append(b_copy)
        best_move = states[random.randint(0, len(states) - 1)]
        self.board = best_move.board
        self.whiteTurn = not best_move.whiteTurn
        self.white = best_move.white
        self.black = best_move.black
        self.blocks = best_move.blocks
      elif turn == 'white' and ai:
        # White turn (AI)
        print('thinking...')
        moves_p = self.genNewMoves(whiteTurn = True)
        states = []
        for move in moves_p:
          b_copy = self.copy()
          b_copy.movePiece(move, 1)
          states.append(b_copy)
        moves_b = self.genBlocks(turn)
        for block in moves_b['h']:
          b_copy = self.copy()
          b_copy.placeBlock(block, 'h')
          b_copy.blocks['white'] -= 1
          states.append(b_copy)
        for block in moves_b['v']:
          b_copy = self.copy()
          b_copy.placeBlock(block, 'v')
          b_copy.blocks['white'] -= 1
          states.append(b_copy)
        best_val = float('-inf')
        best_move = None
        i = 0
        for state in states:
          i+=1
          state_copy = state.copy()
          state_val = state_copy.minimax_value(True, search_depth, float('-inf'), float('inf'))
          if state_val > best_val:
            best_move = state
            best_val = state_val
        self.board = best_move.board
        self.whiteTurn = not best_move.whiteTurn
        self.white = best_move.white
        self.black = best_move.black
        self.blocks = best_move.blocks
      else:
        r = input('(' + turn + ') What would you like to do? (Place Block = B, Move Piece = M): ')
        if r == 'B':
          if self.blocks[turn] > 0:
            r = input('(' + turn + ') Horizontal (h) or vertical (v) or cancel (c)')
            block_lists = self.genBlocks()
            if r == 'c' or (r != 'h' and r != 'v'):
              print('cancelling block')
              continue
            else:
              print('(' + turn + ') Please enter where you want to block from the following list: ')
              print(block_lists[r])
              coord_r = input('(' + turn + ') Enter row: ')
              coord_c = input('(' + turn + ') Enter col: ')
              coord = (int(coord_r), int(coord_c))
              if coord in block_lists[r]:
                self.placeBlock(coord, r)
                print(str(self.blocks['white' if self.whiteTurn else 'black']))
                self.blocks[turn] -= 1
                print(str(self.blocks['white' if self.whiteTurn else 'black']))
                self.whiteTurn = not self.whiteTurn
              else:
                print('(' + turn + ') Input ' + str(coord) + ' is not valid')
                continue
          else:
            print('(' + turn + ') no more blocks left to use for ' + turn)
            continue
        elif r == 'M':
          moves = self.genNewMoves()
          b_copy = self.copy()
          for i in range(len(moves)):
            b_copy.board.bset(moves[i][0], moves[i][1], i)
          print(b_copy.printout())
          r = input('(' + turn + ') select a number to move to from the list above or cancel (c): ')
          if r == 'c':
            print('(' + turn + ') cancelled')
            continue
          try:
            r = int(r)
            self.movePiece(moves[int(r)])
            self.whiteTurn = not self.whiteTurn
          except:
            print('invalid input')
            continue
    print(g.printout())
    print(self.check_win() + ' wins!')

  def heuristic(self, curr = None, whiteTurn = None):
    if whiteTurn == None:
      whiteTurn == self.whiteTurn
    if curr == None:
      curr = self.white if whiteTurn else self.black
    if whiteTurn:
      return curr[0]
    else:
      return self.board.height - 1 - curr[0]

  def eval_function(self, whiteTurn = None):
    if whiteTurn == None:
      whiteTurn = self.whiteTurn
    p_w, l_w = self.path(whiteTurn)
    p_b, l_b = self.path(not whiteTurn)
    if whiteTurn:
      return l_b + self.blocks['white'] - l_w - self.blocks['black']
    else:
      return l_w + self.blocks['black'] - l_b - self.blocks['black']

  # generates minimax value with white as AI and black as player
  def minimax_value(self, white_turn, search_depth, alpha, beta):
    """Return the value of the board, up to the maximum search depth.

    Assumes white is MAX and black is MIN (even if black uses this function).

    Args:
        board (numpy 2D int array) - The othello board
        white_turn (bool) - True iff white would get to play next on the given board
        search_depth (int) - the search depth remaining, decremented for recursive calls
        alpha (int or float) - Lower bound on the value:  MAX ancestor forbids lower results
        beta (int or float) - Upper bound on the value:  MIN ancestor forbids larger results
    """
    winner = self.check_win()
    if search_depth == 0:
      return self.eval_function(white_turn)
    elif winner == 'White':
      return float('inf')
    elif winner == 'Black':
      return float('-inf')
    if white_turn:
      moves_p = self.genNewMoves(whiteTurn = True)
      states = []
      for move in moves_p:
        b_copy = self.copy()
        b_copy.movePiece(move)
        states.append(b_copy)
      moves_b = self.genBlocks(white_turn)
      for block in moves_b['h']:
        b_copy = self.copy()
        b_copy.placeBlock(block, 'h')
        b_copy.blocks['white'] -= 1
        states.append(b_copy)
      for block in moves_b['v']:
        b_copy = self.copy()
        b_copy.placeBlock(block, 'v')
        b_copy.blocks['white'] -= 1
        states.append(b_copy)
      for state in states:
        state.whiteTurn = False
        alpha = max(alpha, state.minimax_value(not white_turn, search_depth - 1, alpha, beta))
        if beta <= alpha:
          break
      return alpha
    else:
      moves_p = self.genNewMoves(whiteTurn = False)
      states = []
      for move in moves_p:
        b_copy = self.copy()
        b_copy.movePiece(move)
        states.append(b_copy)
      moves_b = self.genBlocks(not white_turn)
      for block in moves_b['h']:
        b_copy = self.copy()
        b_copy.placeBlock(block, 'h')
        b_copy.blocks['black'] -= 1
        states.append(b_copy)
      for block in moves_b['v']:
        b_copy = self.copy()
        b_copy.placeBlock(block, 'v')
        b_copy.blocks['black'] -= 1
        states.append(b_copy)
      for state in states:
        state.whiteTurn = True
        beta = min(beta, state.minimax_value(not white_turn, search_depth - 1, alpha, beta))
        if beta <= alpha:
          break
      return beta

g = Game(3, 3, 1)
g.movePiece((1, 1))
l = g.genNewMoves()
for p in l:
  g.board.bset(p[0], p[1], 'M')
print(g.printout())