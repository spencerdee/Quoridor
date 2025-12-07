#ifndef quoridorhpp
#define quoridorhpp

#include "Board.hpp"
#include <vector>
#include <random>

template <int rows = 9, int cols = 9>
class Quoridor
{
    public:
        Quoridor(int blocksPerPlayer = 10)
        {
            this->blocksPerPlayer = blocksPerPlayer;
            whiteBlocks = blocksPerPlayer;
            blackBlocks = blocksPerPlayer;
        };
        ~Quoridor() {};

        void PrintBoard()
        {
            board.PrintBoard();
        };

        bool PlayRandomTurn(bool white = true)
        {
            BoardPosition currentPosition;
            if (white)
            {
                currentPosition = board.whitePosition;
            }
            else
            {
                currentPosition = board.blackPosition;
            }
            std::vector<BoardPosition> validMoves = GenerateValidMoves(currentPosition);
            BoardPosition move = validMoves[rand() % validMoves.size()];
            board.board[currentPosition.row][currentPosition.col] = 0;
            if (white)
            {
                board.whitePosition = move;
                board.board[move.row][move.col] = 'W';
                if (move.row == rows * 2 - 2)
                {
                    printf("WHITE WINS!\n");
                    return true;
                }
                return false;
            }
            else
            {
                board.blackPosition = move;
                board.board[move.row][move.col] = 'B';
                if (move.row == 0)
                {
                    printf("BLACK WINS!\n");
                    return true;
                }
                return false;
            }
        };

        bool PlayHueristicTurn(bool white = true)
        {
            BoardPosition currentPosition;
            if (white)
            {
                currentPosition = board.whitePosition;
            }
            else
            {
                currentPosition = board.blackPosition;
            }
            std::vector<BoardPosition> validMoves = GenerateValidMoves(currentPosition);
            for (const auto& pos : validMoves)
            {
                // Prioritize advancing towards the goal row
                if (white && pos.row > currentPosition.row)
                {
                    BoardPosition move = pos;
                    board.board[currentPosition.row][currentPosition.col] = 0;
                    board.whitePosition = move;
                    board.board[move.row][move.col] = 'W';
                    if (move.row == rows * 2 - 2)
                    {
                        printf("WHITE WINS!\n");
                        return true;
                    }
                    return false;
                }
                else if (!white && pos.row < currentPosition.row)
                {
                    BoardPosition move = pos;
                    board.board[currentPosition.row][currentPosition.col] = 0;
                    board.blackPosition = move;
                    board.board[move.row][move.col] = 'B';
                    if (move.row == 0)
                    {
                        printf("BLACK WINS!\n");
                        return true;
                    }
                    return false;
                }
            }


            board.board[currentPosition.row][currentPosition.col] = 0;
            if (white)
            {
                board.whitePosition = move;
                board.board[move.row][move.col] = 'W';
                if (move.row == rows * 2 - 2)
                {
                    printf("WHITE WINS!\n");
                    return true;
                }
                return false;
            }
            else
            {
                board.blackPosition = move;
                board.board[move.row][move.col] = 'B';
                if (move.row == 0)
                {
                    printf("BLACK WINS!\n");
                    return true;
                }
                return false;
            }
        };

        std::vector<BoardPosition> GenerateValidMoves(BoardPosition position)
        {
            std::vector<BoardPosition> validMoves;
            const int directions[4][2] = {
                {-2, 0}, // up
                {2, 0},  // down
                {0, -2}, // left
                {0, 2}   // right
            };
            for (const auto& dir : directions)
            {
                int newRow = position.row + dir[0];
                int newCol = position.col + dir[1];

                // Check if the new position is within bounds
                if (newRow >= 0 && newRow < rows * 2 - 1 &&
                    newCol >= 0 && newCol < cols * 2 - 1)
                {
                    // Check if there is no wall between current and new position
                    int wallRow = position.row + dir[0] / 2;
                    int wallCol = position.col + dir[1] / 2;
                    if (board.board[wallRow][wallCol] == 0)
                    {
                        validMoves.push_back({newRow, newCol});
                    }
                }
            }
            return validMoves;
        };

    private:
        Board<rows, cols> board;
        int blocksPerPlayer;
        int whiteBlocks;
        int blackBlocks;
};

#endif // quoridorhpp