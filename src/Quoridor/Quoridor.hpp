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

template <int rows = 9, int cols = 9>
class Quoridor
{
    public:
        Quoridor(int blocksPerPlayer = 10)
        {
            this->blocksPerPlayer = blocksPerPlayer;
            whiteBlocks = blocksPerPlayer;
            blackBlocks = blocksPerPlayer;
            srand(static_cast<unsigned int>(std::time(nullptr)));
        };
        ~Quoridor() {};

        void PrintBoard()
        {
            board.PrintBoard();
            printf("White Blocks Remaining: %d\n", whiteBlocks);
            printf("Black Blocks Remaining: %d\n", blackBlocks);
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
                currentPosition = board.whitePosition;
                opponentPosition = board.blackPosition;
            }
            else
            {
                currentPosition = board.blackPosition;
                opponentPosition = board.whitePosition;
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
                currentPosition = board.whitePosition;
                opponentPosition = board.blackPosition;
            }
            else
            {
                currentPosition = board.blackPosition;
                opponentPosition = board.whitePosition;
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

        int GetTreeValue(Turn turn, bool white, uint8_t depth, int alpha, int beta, bool maximizing, bool rootWhite)
        {
            BoardPosition currentPosition;
            BoardPosition opponentPosition;

            if (white)
            {
                currentPosition = board.whitePosition;
                opponentPosition = board.blackPosition;
            }
            else
            {
                currentPosition = board.blackPosition;
                opponentPosition = board.whitePosition;
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
                return AStarSearch(board.whitePosition, true) + whiteBlocks - AStarSearch(board.blackPosition, false) - blackBlocks;
            }
            else
            {
                return AStarSearch(board.blackPosition, false) + blackBlocks - AStarSearch(board.whitePosition, true) - whiteBlocks;
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
                if (newRow >= 0 && newRow < rows * 2 - 1 &&
                    newCol >= 0 && newCol < cols * 2 - 1 &&
                    board.board[position.row + dir[0] / 2][position.col + dir[1] / 2] == 0)
                {
                    if (board.board[newRow][newCol] == 0 || ignorePawns)
                    {
                        validTurns.push_back({BoardPosition{newRow, newCol}, false});
                    }
                    else
                    {
                        // Handle jumping over opponent
                        int jumpRow = newRow + dir[0];
                        int jumpCol = newCol + dir[1];
                        if (jumpRow >= 0 && jumpRow < rows * 2 - 1 &&
                            jumpCol >= 0 && jumpCol < cols * 2 - 1 &&
                            board.board[newRow + dir[0] / 2][newCol + dir[1] / 2] == 0 &&
                            board.board[jumpRow][jumpCol] == 0)
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
                                if (sideRow >= 0 && sideRow < rows * 2 - 1 &&
                                    sideCol >= 0 && sideCol < cols * 2 - 1 &&
                                    board.board[newRow + sideDir[0] / 2][newCol + sideDir[1] / 2] == 0 &&
                                    board.board[sideRow][sideCol] == 0)
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
            if (white && whiteBlocks == 0)
            {
                return;
            }
            else if (!white && blackBlocks == 0)
            {
                return;
            }
            for (int row = 1; row < rows * 2 - 1; row += 2)
            {
                for (int col = 1; col < cols * 2 - 1; col += 2)
                {
                    // Check horizontal placement
                    if (board.board[row][col - 1] == 0 &&
                        board.board[row][col] == 0 &&
                        board.board[row][col + 1] == 0 &&
                        CheckValidPath(true, BlockPosition(row, col, true)) &&
                        CheckValidPath(false, BlockPosition(row, col, true)))
                    {
                        validTurns.push_back(Turn(BlockPosition(row, col, true), true));
                    }
                    // Check vertical placement
                    if (board.board[row - 1][col] == 0 &&
                        board.board[row][col] == 0 &&
                        board.board[row + 1][col] == 0 &&
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
            BoardPosition start = white ? board.whitePosition : board.blackPosition;
            std::set<BoardPosition> visited;

            // Temporarily place the block
            if (blockPos.horizontal)
            {
                board.board[blockPos.row][blockPos.col - 1] = '#';
                board.board[blockPos.row][blockPos.col] = '#';
                board.board[blockPos.row][blockPos.col + 1] = '#';
            }
            else
            {
                board.board[blockPos.row - 1][blockPos.col] = '#';
                board.board[blockPos.row][blockPos.col] = '#';
                board.board[blockPos.row + 1][blockPos.col] = '#';
            }

            bool res = DFS(start, white ? rows * 2 - 2 : 0, visited);

            // Remove the temporary block
            if (blockPos.horizontal)
            {
                board.board[blockPos.row][blockPos.col - 1] = 0;
                board.board[blockPos.row][blockPos.col] = 0;
                board.board[blockPos.row][blockPos.col + 1] = 0;
            }
            else
            {
                board.board[blockPos.row - 1][blockPos.col] = 0;
                board.board[blockPos.row][blockPos.col] = 0;
                board.board[blockPos.row + 1][blockPos.col] = 0;
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
                currentPosition = board.whitePosition;
            }
            else
            {
                currentPosition = board.blackPosition;
            }

            if (move.isBlock)
            {
                BlockPosition block = move.turn.block;
                if (block.horizontal)
                {
                    board.board[block.row][block.col - 1] = '#';
                    board.board[block.row][block.col] = '#';
                    board.board[block.row][block.col + 1] = '#';
                }
                else
                {
                    board.board[block.row - 1][block.col] = '#';
                    board.board[block.row][block.col] = '#';
                    board.board[block.row + 1][block.col] = '#';
                }
                if (white)
                {
                    whiteBlocks--;
                }
                else
                {
                    blackBlocks--;
                }
                return false;
            }
            else
            {
                BoardPosition movePos = move.turn.move;
                if (white)
                {
                    board.board[currentPosition.row][currentPosition.col] = 0;
                    board.whitePosition = movePos;
                    board.board[movePos.row][movePos.col] = 'W';
                    if (movePos.row == rows * 2 - 2)
                    {
                        return true;
                    }
                }
                else
                {
                    board.board[currentPosition.row][currentPosition.col] = 0;
                    board.blackPosition = movePos;
                    board.board[movePos.row][movePos.col] = 'B';
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
                    board.board[block.row][block.col - 1] = 0;
                    board.board[block.row][block.col] = 0;
                    board.board[block.row][block.col + 1] = 0;
                }
                else
                {
                    board.board[block.row - 1][block.col] = 0;
                    board.board[block.row][block.col] = 0;
                    board.board[block.row + 1][block.col] = 0;
                }
                if (white)
                {
                    whiteBlocks++;
                }
                else
                {
                    blackBlocks++;
                }
            }
            else
            {
                // undo move
                if (white)
                {
                    board.board[board.whitePosition.row][board.whitePosition.col] = 0;
                    board.whitePosition = currentPosition;
                    board.board[currentPosition.row][currentPosition.col] = 'W';
                }
                else
                {
                    board.board[board.blackPosition.row][board.blackPosition.col] = 0;
                    board.blackPosition = currentPosition;
                    board.board[currentPosition.row][currentPosition.col] = 'B';
                }
            }
        };

    private:
        Board<rows, cols> board;
        int blocksPerPlayer;
        int whiteBlocks;
        int blackBlocks;

        const int MAX_VALUE = (rows * cols * 2);
        const int MIN_VALUE = -(rows * cols * 2);
};

#endif // quoridorhpp