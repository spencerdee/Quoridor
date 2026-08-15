# TODO
* Optimize current turn processing by taking advantage of caching
    * Current 5x5 board with 3 blocks and depth of 4 takes 21 turns and 111 seconds 
    * Using the Zobrist Hash to represent the game state, and added a compact move struct
    * Next, need to create structs for TT Entry and add updates to tree search function