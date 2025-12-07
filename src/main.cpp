#include <stdio.h>

#include "Quoridor/Quoridor.hpp"

int main(int argc, char *argv[])
{
    Quoridor<9, 9> game = Quoridor<9, 9>(10);
    game.PrintBoard();
    for (int i = 0; i < 10; i++)
    {
        game.PlayRandomTurn(true);
        printf("WHITE MOVE:\n");
        game.PrintBoard();
        game.PlayRandomTurn(false);
        printf("BLACK MOVE:\n");
        game.PrintBoard();
    }
}