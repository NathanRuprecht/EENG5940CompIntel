# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from tkinter import font
import tkinter as tk
import copy
import pickle as pickle
import time

#global variables
root = tk.Tk()
epsilon = 0.9
SIZE = 4
HEIGHT=8
WIDTH=8
FONT = font.Font(family='Courier', size=16, weight=font.BOLD)
NUM_TRAINING = 33554432+1e6
Training = 1
Testing = 1
Playing = 0
results = np.zeros((3,100))

class Game:
    def __init__(self, master, p1, p2, Q_learn=None, Q={}, alpha=0.3, gamma=0.9):
        frame = tk.Frame()
        frame.grid()
        self.master = master
        master.title("Tic Tac Toe")

        self.p1 = p1
        self.p2 = p2
        self.current_player = p1
        self.other_player = p2
        self.empty_text = ""
        self.board = Board()

        self.buttons = [[None for _ in range(SIZE)] for _ in range(SIZE)]
        for i in range(SIZE):
            for j in range(SIZE):
                self.buttons[i][j] = tk.Button(frame, height=HEIGHT, width=WIDTH, text=self.empty_text, 
                            command=lambda i=i, j=j: self.callback(self.buttons[i][j]))
                self.buttons[i][j].grid(row=i, column=j)

        self.reset_button = tk.Button(text="Reset", command=self.reset)
        self.reset_button.grid(row=3)

        self.Q_learn = Q_learn
        if self.Q_learn:
            self.Q = Q
            self.alpha = alpha
            self.gamma = gamma
            self.share_Q_with_players()

    @property
    def Q_learn(self):
        if self._Q_learn is not None:
            return self._Q_learn
        if isinstance(self.p1, Trainer) or isinstance(self.p2, Trainer):
            return True

    @Q_learn.setter
    def Q_learn(self, _Q_learn):
        self._Q_learn = _Q_learn

    def share_Q_with_players(self):
        if isinstance(self.p1, Trainer):
            self.p1.Q = self.Q
        if isinstance(self.p2, Trainer):
            self.p2.Q = self.Q

    def callback(self, button):
        if self.board.over():
            pass                # Do nothing if the game is already over
        else:
            if isinstance(self.current_player, Human) and isinstance(self.other_player, Human):
                if self.empty(button):
                    move = self.get_move(button)
                    self.handle_move(move)
            elif isinstance(self.current_player, Human) and isinstance(self.other_player, Computer):
                computer_player = self.other_player
                if self.empty(button):
                    human_move = self.get_move(button)
                    self.handle_move(human_move)
                    if not self.board.over():
                        computer_move = computer_player.get_move(self.board)
                        self.handle_move(computer_move)

    def empty(self, button):
        return button["text"] == self.empty_text

    def get_move(self, button):
        info = button.grid_info()
        move = (int(info["row"]), int(info["column"]))
        return move

    def handle_move(self, move):
        if self.Q_learn:
            self.learn_Q(move)
        i, j = move
        self.buttons[i][j].configure(text=self.current_player.mark)
        self.board.place_mark(move, self.current_player.mark)
        if self.board.over():
            self.declare_outcome()
        else:
            self.switch_players()

    def declare_outcome(self):
        if self.board.winner() is None:
            #print("Cat's game.")
            pass
        else:
            #print(("The game is over. The player with mark {mark} won!".format(mark=self.current_player.mark)))
            pass

    def reset(self):
        #print("Resetting...")
        for i in range(SIZE):
            for j in range(SIZE):
                self.buttons[i][j].configure(text=self.empty_text)
        self.board = Board(grid=np.ones((SIZE,SIZE))*np.nan)
        self.current_player = self.p1
        self.other_player = self.p2
        self.play()

    def switch_players(self):
        if self.current_player == self.p1:
            self.current_player = self.p2
            self.other_player = self.p1
        else:
            self.current_player = self.p1
            self.other_player = self.p2

    def play(self):
        if isinstance(self.p1, Human) and isinstance(self.p2, Human):
            pass
        elif isinstance(self.p1, Human) and isinstance(self.p2, Computer):
            pass
        elif isinstance(self.p1, Computer) and isinstance(self.p2, Human):
            first_computer_move = self.p1.get_move(self.board)
            self.handle_move(first_computer_move)
        elif isinstance(self.p1, Computer) and isinstance(self.p2, Computer):
            while not self.board.over():
                self.play_turn()

    def play_turn(self):
        move = self.current_player.get_move(self.board)
        self.handle_move(move)

    def learn_Q(self, move):
        state_key = Trainer.make_and_maybe_add_key(self.board, self.current_player.mark, self.Q)
        next_board = self.board.get_next_board(move, self.current_player.mark)
        reward = next_board.give_reward()
        next_state_key = Trainer.make_and_maybe_add_key(next_board, self.other_player.mark, self.Q)
        if next_board.over():
            expected = reward
        else:
            next_Qs = self.Q[next_state_key]
            if self.current_player.mark == "X":
                expected = reward + (self.gamma * min(next_Qs.values()))
            elif self.current_player.mark == "O":
                expected = reward + (self.gamma * max(next_Qs.values()))
        change = self.alpha * (expected - self.Q[state_key][move])
        self.Q[state_key][move] += change


class Board:
    def __init__(self, grid=np.ones((SIZE,SIZE))*np.nan):
        self.grid = grid

    def winner(self):
        rows = [self.grid[i,:] for i in range(SIZE)]
        cols = [self.grid[:,j] for j in range(SIZE)]
        diag = [np.array([self.grid[i,i] for i in range(SIZE)])]
        cross_diag = [np.array([self.grid[2-i,i] for i in range(SIZE)])]
        lanes = np.concatenate((rows, cols, diag, cross_diag))

        any_lane = lambda x: any([np.array_equal(lane, x) for lane in lanes])
        if any_lane(np.ones(SIZE)):
            return "X"
        elif any_lane(np.zeros(SIZE)):
            return "O"

    def over(self):
        return (not np.any(np.isnan(self.grid))) or (self.winner() is not None)

    def place_mark(self, move, mark):
        num = Board.mark2num(mark)
        self.grid[tuple(move)] = num

    @staticmethod
    def mark2num(mark):
        d = {"X": 1, "O": 0}
        return d[mark]

    def available_moves(self):
        return [(i,j) for i in range(SIZE) for j in range(SIZE) if np.isnan(self.grid[i][j])]

    def get_next_board(self, move, mark):
        next_board = copy.deepcopy(self)
        next_board.place_mark(move, mark)
        return next_board

    def make_key(self, mark):
        fill_value = 9
        filled_grid = copy.deepcopy(self.grid)
        np.place(filled_grid, np.isnan(filled_grid), fill_value)
        return "".join(map(str, (list(map(int, filled_grid.flatten()))))) + mark

    def give_reward(self):
        if self.over():
            if self.winner() is not None:
                if self.winner() == "X":
                    return 1.0
                elif self.winner() == "O":
                    return -1.0
            else:
                return 0.5
        else:
            return 0.0


class Player(object):
    def __init__(self, mark):
        self.mark = mark

    @property
    def opponent_mark(self):
        if self.mark == 'X':
            return 'O'
        elif self.mark == 'O':
            return 'X'
        else:
            print("The player's mark must be either 'X' or 'O'.")

class Human(Player):
    pass

class Computer(Player):
    pass

class Trainer(Computer):
    def __init__(self, mark, Q={}, epsilon=0.2):
        super(Trainer, self).__init__(mark=mark)
        self.Q = Q
        self.epsilon = epsilon

    def get_move(self, board):
        if np.random.uniform() < self.epsilon: #Exploration!              
            moves = board.available_moves()
            if moves:
                return moves[np.random.choice(len(moves))]
        else: #No exploration...
            state_key = Trainer.make_and_maybe_add_key(board, self.mark, self.Q)
            Qs = self.Q[state_key]

            if self.mark == "X":
                return Trainer.stochastic_argminmax(Qs, max)
            elif self.mark == "O":
                return Trainer.stochastic_argminmax(Qs, min)

    @staticmethod
    def make_and_maybe_add_key(board, mark, Q):
        default_Qvalue = 1.0
        state_key = board.make_key(mark)
        if Q.get(state_key) is None:
            moves = board.available_moves()
            Q[state_key] = {move: default_Qvalue for move in moves}
        return state_key

    @staticmethod
    def stochastic_argminmax(Qs, min_or_max):
        min_or_maxQ = min_or_max(list(Qs.values()))
        if list(Qs.values()).count(min_or_maxQ) > 1:
            best_options = [move for move in list(Qs.keys()) if Qs[move] == min_or_maxQ]
            move = best_options[np.random.choice(len(best_options))]
        else:
            move = min_or_max(Qs, key=Qs.get)
        return move

def main():
    global root, Training, Testing, results, NUM_TRAINING
    Q=pickle.load(open("Q_final_{}.p".format(int(NUM_TRAINING-int(1e6))), "rb"))
    
    while True:
        current = int(NUM_TRAINING)
        previous = int(NUM_TRAINING - 1e6)
        if Training:
            if current != 33554432:
                Q = pickle.load(open("Q_final_{}.p".format(previous), "rb"))
            p1 = Trainer(mark="X",epsilon = epsilon)
            p2 = Trainer(mark="O",epsilon = epsilon)
            game = Game(root, p1, p2)
        
            start = time.time()
            for i in range(previous, current):
                print(i, "out of", current)
                game.play()
                game.reset()
            end = time.time()-start
            Q = game.Q
            pickle.dump(Q, open("Q_final_{}.p".format(current), "wb"))
    
        if Testing:
            Q = game.Q
           
            p1 = Trainer(mark="X", epsilon=0)
            p2 = Trainer(mark="O", epsilon=0)
            game = Game(root, p1, p2, Q=Q)
            game.reset()
        
            for i in range(100):
                #print("Testing game", i)
                game.play()
            
                if game.board.winner() is None:
                    results[0,i] = 1
                else:
                    results[0,i] = 0
            
                if game.board.winner() is "X":
                    results[1,i] = 1
                else:
                    results[1,i] = 0
            
                if game.board.winner() is "O":
                    results[2,i] = 1
                else:
                    results[2,i] = 0
                
                game.reset()
                
            f = open("Results.txt", "r")
            contents = f.read()
            f.close()
            
            f = open("Results.txt", "w")
            f.write(contents)
            contents = "Trial %d.\tTime %.4f.\tCats %i.\tP1 %i.\tP2 %i\r"% (NUM_TRAINING, end/60, np.sum(results[0,:]), np.sum(results[1,:]), np.sum(results[2,:]))
            print(contents)
            f.write(contents)
            f.close()

            NUM_TRAINING = NUM_TRAINING+1e6
            
    if Playing:
        print("Playing")
        Q = pickle.load(open("Q_final_8388608.p", "rb"))
        print("Q loaded")
        
        p1 = Human(mark="X")
        p2 = Trainer(mark="O", epsilon=0)
        print("Players ready... making game")
        game = Game(root, p1, p2, Q=Q)

        game.play()
        root.mainloop()

#Window main parent and title
if __name__ == "__main__":
    main()
