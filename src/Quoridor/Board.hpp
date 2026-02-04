#ifndef boardhpp
#define boardhpp

#include <cstring>

struct BoardPosition
{
    int row;
    int col;

    bool operator==(const BoardPosition& other) const
    {
        return row == other.row && col == other.col;
    }

    bool operator<(const BoardPosition& other) const
    {
        return (row < other.row) || (row == other.row && col < other.col);
    }
};

struct BlockPosition
{
    int row;
    int col;
    bool horizontal;
};

struct Turn
{
    union
    {
        BoardPosition move;
        BlockPosition block;
    } turn;
    bool isBlock;

    Turn(BoardPosition position, bool isBlock)
    {
        turn.move = position;
        this->isBlock = isBlock;
    };

    Turn(BlockPosition blockPosition, bool isBlock)
    {
        turn.block = blockPosition;
        this->isBlock = isBlock;
    };

    Turn() {};
};

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
};

#endif // board