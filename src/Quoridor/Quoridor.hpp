#ifndef quoridorhpp
#define quoridorhpp

#include "Board.hpp"
#include <climits>
#include <queue>
#include <vector>
#include <random>
#include <ctime>
#include <map>
#include <set>
#include <stdint.h>
#include <stdexcept>

template <uint8_t rows = 9, uint8_t cols = 9, uint8_t blocksPerPlayer = 10>
class Quoridor
{
    public:
        Quoridor()
        {
            static_assert(rows < 10 && cols < 10, "Rows and columns must be less than 10");
            static_assert(blocksPerPlayer < 10, "Number of blocks per player must be less than 10");

            gameState.whiteBlocks = blocksPerPlayer;
            gameState.blackBlocks = blocksPerPlayer;
            srand(static_cast<unsigned int>(std::time(nullptr)));

            InitializeZobristTable();
        };
        ~Quoridor() {};

        void PrintBoard()
        {
            gameState.PrintBoard();
            printf("White Blocks Remaining: %d\n", gameState.whiteBlocks);
            printf("Black Blocks Remaining: %d\n", gameState.blackBlocks);
        };

        bool PlayRandomTurn(bool white = true)
        {
            BoardPosition currentPosition;
            if (white)
            {
                currentPosition = gameState.whitePosition;
            }
            else
            {
                currentPosition = gameState.blackPosition;
            }
            std::vector<Turn> validTurns;
            GenerateValidMoves(currentPosition, validTurns);
            GenerateValidBlocks(white, validTurns);
            Turn turn = validTurns[rand() % validTurns.size()];
            return PlayMove(turn, white);
        };

        bool PlayHueristicTurn(bool white = true)
        {
            BoardPosition currentPosition;
            BoardPosition opponentPosition;
            if (white)
            {
                currentPosition = gameState.whitePosition;
                opponentPosition = gameState.blackPosition;
            }
            else
            {
                currentPosition = gameState.blackPosition;
                opponentPosition = gameState.whitePosition;
            }
            std::vector<Turn> validTurns;
            GenerateValidMoves(currentPosition, validTurns);
            GenerateValidBlocks(white, validTurns);

            // select the move with the lowest cost
            int lowestCost = INT_MAX;
            Turn bestMove = {};
            for (const auto& pos : validTurns)
            {
                PlayMove(pos, white);
                int cost;
                if (pos.isBlock)
                {
                    cost = AStarSearch(currentPosition, white) - AStarSearch(opponentPosition, !white);
                }
                else
                {
                    cost = AStarSearch(pos.turn.move, white) - AStarSearch(opponentPosition, !white);
                }
                
                UndoMove(pos, white, currentPosition);
                if (cost <= lowestCost)
                {
                    lowestCost = cost;
                    bestMove = pos;
                }
            }

            return PlayMove(bestMove, white);
        };

        bool PlayTreeMove(bool white = true, uint8_t depth = 2)
        {
            if (depth < 1)
            {
                return PlayHueristicTurn(white);
            }

            BoardPosition currentPosition;
            BoardPosition opponentPosition;
            if (white)
            {
                currentPosition = gameState.whitePosition;
                opponentPosition = gameState.blackPosition;
            }
            else
            {
                currentPosition = gameState.blackPosition;
                opponentPosition = gameState.whitePosition;
            }
            std::vector<Turn> validTurns;
            GenerateValidMoves(currentPosition, validTurns);
            GenerateValidBlocks(white, validTurns);

            int bestValue = INT_MIN;
            Turn bestTurn = {};
            for (const auto& turn : validTurns)
            {
                int value = GetTreeValue(turn, white, depth, INT_MIN, INT_MAX, false, white);
                if (value > bestValue)
                {
                    bestValue = value;
                    bestTurn = turn;
                }

                if (value >= MAX_VALUE / 2)
                {
                    break;
                }
            }
            return PlayMove(bestTurn, white);
        };

        bool PlayMonteCarloMove(bool white = true, int simulations = 1000)
        {
            // generate valid moves at current state
            std::vector<Turn> validTurns;
            GenerateValidMoves(white ? gameState.whitePosition : gameState.blackPosition, validTurns);
            GenerateValidBlocks(white, validTurns);

            for (int i = 0; i < simulations; i++)
            {
                int bestValue = -MAX_VALUE;
                // selection - select best child based on hueristic
                for (const auto& turn : validTurns)
                {
                    PlayMove(turn, white);
                    int cost = AStarSearch(white ? gameState.whitePosition : gameState.blackPosition, white) - AStarSearch(white ? gameState.blackPosition : gameState.whitePosition, !white);
                    UndoMove(turn, white);
                    if (cost >= MAX_VALUE)
                    {
                        return PlayMove(turn, white);
                    }
                }
            }
        }



        int GetTreeValue(Turn turn, bool white, uint8_t depth, int alpha, int beta, bool maximizing, bool rootWhite)
        {
            BoardPosition currentPosition;
            BoardPosition opponentPosition;

            if (white)
            {
                currentPosition = gameState.whitePosition;
                opponentPosition = gameState.blackPosition;
            }
            else
            {
                currentPosition = gameState.blackPosition;
                opponentPosition = gameState.whitePosition;
            }

            if (depth == 0)
            {
                int value;
                if (PlayMove(turn, white))
                {
                    int value = ((MAX_VALUE - AStarSearch(opponentPosition, !white)) * ((white == rootWhite) * 2 - 1));
                }
                else
                {
                    value = GetMinimaxCost(rootWhite);
                }

                UndoMove(turn, white, currentPosition);
                
                return value;
            }
            else if (depth > 0)
            {
                if (PlayMove(turn, white))
                {
                    int value = ((MAX_VALUE - AStarSearch(opponentPosition, !white)) * ((rootWhite == white) * 2 - 1));
                    UndoMove(turn, white, currentPosition);
                    return value;
                }

                // generate opponent's next moves
                std::vector<Turn> validTurns;
                GenerateValidMoves(opponentPosition, validTurns);
                GenerateValidBlocks(!white, validTurns);

                if (!maximizing)
                {
                    int bestValue = MAX_VALUE * 2; 
                    for (const auto& nextTurn : validTurns)
                    {
                        int value = GetTreeValue(nextTurn, !white, depth - 1, alpha, beta, !maximizing, rootWhite);
                        bestValue = std::min(value, bestValue);
                        if (bestValue <= alpha)
                        {
                            break;
                        }
                        beta = std::min(beta, bestValue);
                    }
                    UndoMove(turn, white, currentPosition);

                    return bestValue;
                }
                else
                {
                    int bestValue = MIN_VALUE * 2; 
                    Turn bestTurn = {};
                    for (const Turn& nextTurn : validTurns)
                    {
                        int value = GetTreeValue(nextTurn, !white, depth - 1, alpha, beta, !maximizing, rootWhite);
                        if (value > bestValue)
                        {
                            bestTurn = nextTurn;
                        }
                        bestValue = std::max(value, bestValue);
                        if (bestValue >= beta)
                        {
                            break;
                        }
                        alpha = std::max(alpha, bestValue);
                    }
                    UndoMove(turn, white, currentPosition);

                    return bestValue;
                }
            }

            throw std::invalid_argument("Invalid depth");
        };

        int GetMinimaxCost(bool white)
        {
            if (!white)
            {
                return AStarSearch(gameState.whitePosition, true) + gameState.whiteBlocks - AStarSearch(gameState.blackPosition, false) - gameState.blackBlocks;
            }
            else
            {
                return AStarSearch(gameState.blackPosition, false) + gameState.blackBlocks - AStarSearch(gameState.whitePosition, true) - gameState.whiteBlocks;
            }
        };

        void GenerateValidMoves(BoardPosition position, std::vector<Turn>& validTurns, bool ignorePawns = false)
        {
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

                // Check if the new position is within bounds and there is no wall blocking the way
                if (newRow >= 0 && newRow < numWallRows &&
                    newCol >= 0 && newCol < numWallCols &&
                    gameState.board[position.row + dir[0] / 2][position.col + dir[1] / 2] == 0)
                {
                    if (gameState.board[newRow][newCol] == 0 || ignorePawns)
                    {
                        validTurns.push_back({BoardPosition{newRow, newCol}, false});
                    }
                    else
                    {
                        // Handle jumping over opponent
                        int jumpRow = newRow + dir[0];
                        int jumpCol = newCol + dir[1];
                        if (jumpRow >= 0 && jumpRow < numWallRows &&
                            jumpCol >= 0 && jumpCol < numWallCols &&
                            gameState.board[newRow + dir[0] / 2][newCol + dir[1] / 2] == 0 &&
                            gameState.board[jumpRow][jumpCol] == 0)
                        {
                            validTurns.push_back({BoardPosition{jumpRow, jumpCol}, false});
                        }
                        else
                        {
                            // Check for side jumps
                            const int sideDirs[2][2] = {
                                { -dir[1], -dir[0] }, // left side
                                { dir[1], dir[0] }    // right side
                            };
                            for (const auto& sideDir : sideDirs)
                            {
                                int sideRow = newRow + sideDir[0];
                                int sideCol = newCol + sideDir[1];
                                if (sideRow >= 0 && sideRow < numWallRows &&
                                    sideCol >= 0 && sideCol < numWallCols &&
                                    gameState.board[newRow + sideDir[0] / 2][newCol + sideDir[1] / 2] == 0 &&
                                    gameState.board[sideRow][sideCol] == 0)
                                {
                                    validTurns.push_back({BoardPosition{sideRow, sideCol}, false});
                                }
                            }
                        }
                    }
                }
            }
        };

        void GenerateValidBlocks(bool white, std::vector<Turn>& validTurns)
        {
            if (white && gameState.whiteBlocks == 0)
            {
                return;
            }
            else if (!white && gameState.blackBlocks == 0)
            {
                return;
            }
            for (int row = 1; row < numWallRows; row += 2)
            {
                for (int col = 1; col < numWallCols; col += 2)
                {
                    // Check horizontal placement
                    if (gameState.board[row][col - 1] == 0 &&
                        gameState.board[row][col] == 0 &&
                        gameState.board[row][col + 1] == 0 &&
                        CheckValidPath(true, BlockPosition(row, col, true)) &&
                        CheckValidPath(false, BlockPosition(row, col, true)))
                    {
                        validTurns.push_back(Turn(BlockPosition(row, col, true), true));
                    }
                    // Check vertical placement
                    if (gameState.board[row - 1][col] == 0 &&
                        gameState.board[row][col] == 0 &&
                        gameState.board[row + 1][col] == 0 &&
                        CheckValidPath(true, BlockPosition(row, col, false)) &&
                        CheckValidPath(false, BlockPosition(row, col, false)))
                    {
                        validTurns.push_back({BlockPosition(row, col, false), true});
                    }
                }
            }
        };

        bool CheckValidPath(bool white, BlockPosition blockPos)
        {
            BoardPosition start = white ? gameState.whitePosition : gameState.blackPosition;
            std::set<BoardPosition> visited;

            // Temporarily place the block
            if (blockPos.horizontal)
            {
                gameState.board[blockPos.position.row][blockPos.position.col - 1] = '#';
                gameState.board[blockPos.position.row][blockPos.position.col] = '#';
                gameState.board[blockPos.position.row][blockPos.position.col + 1] = '#';
            }
            else
            {
                gameState.board[blockPos.position.row - 1][blockPos.position.col] = '#';
                gameState.board[blockPos.position.row][blockPos.position.col] = '#';
                gameState.board[blockPos.position.row + 1][blockPos.position.col] = '#';
            }

            bool res = DFS(start, white ? rows * 2 - 2 : 0, visited);

            // Remove the temporary block
            if (blockPos.horizontal)
            {
                gameState.board[blockPos.position.row][blockPos.position.col - 1] = 0;
                gameState.board[blockPos.position.row][blockPos.position.col] = 0;
                gameState.board[blockPos.position.row][blockPos.position.col + 1] = 0;
            }
            else
            {
                gameState.board[blockPos.position.row - 1][blockPos.position.col] = 0;
                gameState.board[blockPos.position.row][blockPos.position.col] = 0;
                gameState.board[blockPos.position.row + 1][blockPos.position.col] = 0;
            }

            return res;
        };

        bool DFS(BoardPosition position, int goalRow, std::set<BoardPosition>& visited)
        {
            if (position.row == goalRow)
            {
                return true;
            }

            visited.insert(position);

            std::vector<Turn> neighbors;
            GenerateValidMoves(position, neighbors);

            for (const auto& neighbor : neighbors)
            {
                if (visited.contains(neighbor.turn.move))
                {
                    continue;
                }
                else
                {
                    if (DFS(neighbor.turn.move, goalRow, visited))
                    {
                        return true;
                    }
                }
            }

            return false;
        };

        int CalculateHeuristic(BoardPosition a, bool white)
        {
            if (white)
            {
                return (rows * 2 - 2) - a.row;
            }
            else
            {
                return a.row;
            }
        };

        int AStarSearch(BoardPosition start, bool white)
        {
            std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> openSet;
            std::map<BoardPosition, AStarNode> visitedNodes;

            AStarNode startNode = AStarNode(start, 0, CalculateHeuristic(start, white), nullptr);
            openSet.push(startNode);
            visitedNodes[start] = startNode;

            while (!openSet.empty()) {
                AStarNode current = openSet.top();
                openSet.pop();

                if (current.hCost == 0) {
                    return current.gCost;
                }

                visitedNodes[current.position] = current;

                // Explore neighbors and update costs
                std::vector<Turn> validTurns;
                GenerateValidMoves(current.position, validTurns, true);
                for (const auto& pos : validTurns)
                {
                    AStarNode neighbor = AStarNode(pos.turn.move, current.gCost + 2, CalculateHeuristic(pos.turn.move, white), &visitedNodes[current.position]);
                    if (!visitedNodes.contains(pos.turn.move))
                    {
                        openSet.push(neighbor);
                    }
                    else if (visitedNodes[pos.turn.move].gCost > neighbor.gCost)
                    {
                        visitedNodes[pos.turn.move].gCost = neighbor.gCost;
                        visitedNodes[pos.turn.move].parent = &visitedNodes[current.position];
                        openSet.push(visitedNodes[pos.turn.move]);
                    }
                }
            }

            return INT_MAX; // Return an invalid node if no path is found
        };

        bool PlayMove(Turn move, bool white)
        {
            BoardPosition currentPosition;
            if (white)
            {
                currentPosition = gameState.whitePosition;
            }
            else
            {
                currentPosition = gameState.blackPosition;
            }

            if (move.isBlock)
            {
                BlockPosition block = move.turn.block;
                if (block.horizontal)
                {
                    gameState.board[block.position.row][block.position.col - 1] = '#';
                    gameState.board[block.position.row][block.position.col] = '#';
                    gameState.board[block.position.row][block.position.col + 1] = '#';
                }
                else
                {
                    gameState.board[block.position.row - 1][block.position.col] = '#';
                    gameState.board[block.position.row][block.position.col] = '#';
                    gameState.board[block.position.row + 1][block.position.col] = '#';
                }
                if (white)
                {
                    gameState.whiteBlocks--;
                }
                else
                {
                    gameState.blackBlocks--;
                }
                return false;
            }
            else
            {
                BoardPosition movePos = move.turn.move;
                if (white)
                {
                    gameState.board[currentPosition.row][currentPosition.col] = 0;
                    gameState.whitePosition = movePos;
                    gameState.board[movePos.row][movePos.col] = 'W';
                    if (movePos.row == rows * 2 - 2)
                    {
                        return true;
                    }
                }
                else
                {
                    gameState.board[currentPosition.row][currentPosition.col] = 0;
                    gameState.blackPosition = movePos;
                    gameState.board[movePos.row][movePos.col] = 'B';
                    if (movePos.row == 0)
                    {
                        return true;
                    }
                }
            }
            return false;
        };

        void UndoMove(Turn move, bool white, BoardPosition currentPosition = {})
        {
            if (move.isBlock)
            {
                // undo block placement
                BlockPosition block = move.turn.block;
                if (block.horizontal)
                {
                    gameState.board[block.position.row][block.position.col - 1] = 0;
                    gameState.board[block.position.row][block.position.col] = 0;
                    gameState.board[block.position.row][block.position.col + 1] = 0;
                }
                else
                {
                    gameState.board[block.position.row - 1][block.position.col] = 0;
                    gameState.board[block.position.row][block.position.col] = 0;
                    gameState.board[block.position.row + 1][block.position.col] = 0;
                }
                if (white)
                {
                    gameState.whiteBlocks++;
                }
                else
                {
                    gameState.blackBlocks++;
                }
            }
            else
            {
                // undo move
                if (white)
                {
                    gameState.board[gameState.whitePosition.row][gameState.whitePosition.col] = 0;
                    gameState.whitePosition = currentPosition;
                    gameState.board[currentPosition.row][currentPosition.col] = 'W';
                }
                else
                {
                    gameState.board[gameState.blackPosition.row][gameState.blackPosition.col] = 0;
                    gameState.blackPosition = currentPosition;
                    gameState.board[currentPosition.row][currentPosition.col] = 'B';
                }
            }
        };

    private:
        constexpr static int numWallRows = (rows * 2 - 1);
        constexpr static int numWallCols = (cols * 2 - 1);

        struct ZobristTable {
            // 2 Players, with rows x cols positions for pawns
            uint64_t pawn_positions[2][rows][cols];

            // 2 Orientations (Horizontal=1, Vertical=0)
            uint64_t walls[2][numWallRows][numWallCols];

            // 2 Players, with blocksPerPlayer possible wall counts (0 through blocksPerPlayer)
            uint64_t wall_inventory[2][blocksPerPlayer + 1];

            // Applied to the hash if it is whites turn to move, otherwise it is black's turn to move
            uint64_t white_to_move;
        };

        void InitializeZobristTable()
        {
            // 1. Set up the random number generator
            std::mt19937_64 rng(0x12345678); 
    
            // 2. Set up the distribution to cover the full 64-bit range
            std::uniform_int_distribution<uint64_t> dist;

            // 3. Fill the Zobrist table with random numbers
            for (int player = 0; player < 2; ++player) {
                for (int row = 0; row < rows; ++row) {
                    for (int col = 0; col < cols; ++col) {
                        zobristTable.pawn_positions[player][row][col] = dist(rng);
                    }
                }
            }

            for (int orientation = 0; orientation < 2; ++orientation) {
                for (int row = 0; row < numWallRows; ++row) {
                    for (int col = 0; col < numWallCols; ++col) {
                        zobristTable.walls[orientation][row][col] = dist(rng);
                    }
                }
            }

            for (int player = 0; player < 2; ++player) {
                for (int count = 0; count <= blocksPerPlayer; ++count) {
                    zobristTable.wall_inventory[player][count] = dist(rng);
                }
            }

            zobristTable.white_to_move = dist(rng);
        };

        // The current gamestate
        Board<rows, cols> gameState;

        // Define the maximum and minimum values for the evaluation function
        // Set to twice the total number of cells on the board to ensure that any 
        // generated path on the board will be smaller than these values
        const int MAX_VALUE = (rows * cols * 2);
        const int MIN_VALUE = -(rows * cols * 2);

        ZobristTable zobristTable;
};

#endif // quoridorhpp