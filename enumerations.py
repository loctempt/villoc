from enum import Enum

class InstStat(Enum):
    OK = 0
    UAF_DANGLING = -4
    UAF_BASIC = -3
    IDLE = -2
    ERR = -1
    UAF = "Use After Free"
    OVF = "Heap Overflow"

class UaddrStat(Enum):
    INIT = 0
    INUSE = 1
    FREED = -1  # 暂未使用

class PointerStat(Enum):
    VALID = 0
    DANGLING = -1