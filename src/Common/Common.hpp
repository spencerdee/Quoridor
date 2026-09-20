#ifndef commonhpp
#define commonhpp

#define PACKED __attribute__((packed))

#define DBUG(x, ...) printf("[DEBUG] " x "\n", ##__VA_ARGS__);

#endif