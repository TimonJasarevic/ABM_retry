from enum import Enum


class NetworkType(Enum):
    ScaleFree = 1
    Random = 2
    Regular = 3


class MessageState(Enum):
    Unaware = 1
    FalseBeliever = 2  # unverified false forward; a display label, not knowledge
    TrueBeliever = 3   # true forward, verified or not
    Corrected = 4
    Discarded = 5


class Action(Enum):
    VERIFY = 1
    SHARE = 2
    DISCARD = 3
