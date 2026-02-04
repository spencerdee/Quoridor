#include <stdio.h>
#include <chrono>

#include "Quoridor/Quoridor.hpp"

int main(int argc, char *argv[])
{
    Quoridor<5, 5> game = Quoridor<5, 5>(3);
    game.PrintBoard();

    int turns = 0;
    bool whiteTurn = true;
    auto start = std::chrono::high_resolution_clock::now();
    while (true)
    {
        if (game.PlayTreeMove(whiteTurn, 4))
        {
            printf("White Player Wins!\n");
            break;
        }
        game.PrintBoard();
        printf("\n\nTurn: %d\n", turns);
        turns++;

        if (game.PlayTreeMove(!whiteTurn, 3))
        {
            printf("Black Player Wins!\n");
            break;
        }
        game.PrintBoard();
        printf("\n\nTurn: %d\n", turns);
        turns++;
    }
    game.PrintBoard();
    auto end =  std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    printf("Game finished in %d turns and %lld seconds\n", turns, duration);
}