#ifndef boardhpp
#define boardhpp

#include <cstring>
#include "Common.hpp"

/**
 * @brief Represents a position on the board with row and column coordinates.
 */
struct BoardPosition
{
    uint8_t row;
    uint8_t col;

    bool operator==(const BoardPosition& other) const
    {
        return row == other.row && col == other.col;
    }

    bool operator<(const BoardPosition& other) const
    {
        return (row < other.row) || (row == other.row && col < other.col);
    }
};

/**
 * @brief Represents a position of a block on the board.
 * @details The block can be placed either horizontally or vertically, indicated by the `horizontal` boolean.
 */
struct BlockPosition
{
    BoardPosition position;
    bool horizontal : 1;
};

/**
 * @brief Represents a turn in the game, which can be either a move or a block placement.
 */
struct PACKED Turn
{
    uint8_t row : 7;
    uint8_t col : 7;
    bool horizontal : 1;
    bool isBlock : 1;

    Turn(BoardPosition position)
    {
        this->row = position.row;
        this->col = position.col;
        this->horizontal = false;
        this->isBlock = false;
    };

    Turn(BlockPosition blockPosition)
    {
        this->row = blockPosition.position.row;
        this->col = blockPosition.position.col;
        this->horizontal = blockPosition.horizontal;
        this->isBlock = true;
    };

    Turn() {};
};

/**
 * @brief Represents a node in the A* search algorithm.
 */
struct AStarNode
{
    BoardPosition position;
    int gCost; // Cost from start to current node
    int hCost; // Heuristic cost from current node to goal
    int fCost() const { return gCost + hCost; } // Total cost
    AStarNode* parent; // Pointer to parent node for path reconstruction

    bool operator>(const AStarNode& other) const
    {
        return fCost() > other.fCost();
    }

    bool operator<(const AStarNode& other) const
    {
        return fCost() < other.fCost();
    }
};

template <int rows = 9, int cols = 9>
class Board
{
    public:
        Board()
        {
            for (int i = 0; i < rows * 2 - 1; i++)
            {
                memset(board[i], 0, sizeof(board[i]));
            }

            // set the initial positions of the players
            board[whitePosition.row][whitePosition.col] = 'W';
            board[blackPosition.row][blackPosition.col] = 'B';
        };
        ~Board() {};

        void PrintBoard()
        {
            printf("  ");
            for (int i = 0; i < cols * 2 - 1; i++)
            {
                printf("%d ", i % 10);
            }

            printf("\n");
            for (int i = 0; i < rows * 2 - 1; i++)
            {
                printf("%d ", i % 10);

                for (int j = 0; j < cols * 2 - 1; j++)
                {
                    if (board[i][j] == 0)
                    {
                        if (i % 2 == 0 && j % 2 == 0)
                        {
                            printf("* ");
                        }
                        else
                        {
                            printf("` ");
                        }
                    }
                    else
                    {
                        printf("%c ", board[i][j]);
                    }
                }
                printf("\n");
            }
        };

        // board representation, for a 3x3 board:
        // S B S B S
        // B B B B B
        // S B S B S
        // B B B B B
        // S B S B S
        char board[rows * 2 - 1][cols * 2 - 1];

        BoardPosition whitePosition = {0, cols - 1};
        BoardPosition blackPosition = {rows * 2 - 2, cols - 1};

        uint whiteBlocks = 0;
        uint blackBlocks = 0;
};

#endif // board