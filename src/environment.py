from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Any
from base import Scene, Event, ReviewEvent, AttributeEvent

class Language_Observation(ABC):
    @abstractmethod
    def to_sequence(self) -> Tuple[List[Tuple[str, Optional[float]]], bool]:
        # returns a List of Tuples and a bool indicating terminal
        # each review Tuple should be: (str, None) (state)
        # each atrribute Tuple should be: (str, reward) (action)
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

class RecObservation(Language_Observation):
    def __init__(self, scene:Scene, event:Optional[Event]=None):
        self.scene = scene
        self.event = event

    def to_sequence(self) -> Tuple[List[Tuple[str, float | None]], bool]:
        # have to update this function with reward
        if self.event is None:
            return [(self.scene.init_state)], False
        evs = self.event.get_events() # event is [-1] item, get_events() call events reverse order and sort back
        sequence = [self.scene.init_state]
        sequence += [str(evs[i]) for i in range(len(evs))]
        # have to set terminal as well
        terminal = True
        return sequence, terminal

    def __str__(self) ->str :
        if self.event is None:
            return self.scene.init_state
        return self.scene.init_state+'\n'+'\n'.join(list(map(str, self.event.get_events())))