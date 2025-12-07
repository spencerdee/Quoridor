#ifndef boardhpp
#define boardhpp

#include <cstring>

struct BoardPosition
{
    int row;
    int col;
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
            for (int i = 0; i < cols; i++)
            {
                printf("%d   ", i);
            }

            printf("\n");
            for (int i = 0; i < rows * 2 - 1; i++)
            {
                if (i % 2 == 0)
                {
                    printf("%d ", i / 2);
                }
                else
                {
                    printf("  ");
                }

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