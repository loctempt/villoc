from enum import Enum

class InstStat(Enum):
    OK = 0
    IDLE = -2
    ERR = -1
    UAF = 1
    OVF = 2