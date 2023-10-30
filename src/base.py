from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import json

@dataclass
class Event:
    def append(self, ev:Event, link_forward=False):
        ev.prev = self
        if link_forward:
            self.next = ev
        ev.scene = self.scene # which scene that event is included in 
        return ev

    def get_events(self, direction='prev'):
        if direction == 'prev':
            func = lambda ev: ev.prev
        elif direction == 'next':
            func = lambda ev: ev.next
        else:
            raise NotImplementedError
        events = []
        ev = self
        while ev is not None:
            events.append(ev)
            ev = func(ev)
        if direction == 'prev':
            events.reverse()
        return events

    def is_final(self):
        return isinstance(self, StopEvent)

@dataclass
class AttributeEvent(Event): # question = action = attribute
    attribute : str
    #progress : float # related to reward? 
    scene : Scene # which is (self) involved in 
    prev : Optional[Event]
    next : Optional[Event]

    def __str__(self):
        return self.attribute

@dataclass
class ReviewEvent(Event): # answer = env = review
    review : str
    #reward : float
    #progress : float
    scene : Scene
    prev : Optional[Event]
    next : Optional[Event]

    def __str__(self):
        return self.review

@dataclass
class StopEvent(Event):
    #progress : float
    scene : Scene
    prev : Optional[Event]
    next : Optional[Event]

    def __str__(self):
        return '<stop>'

@dataclass
class Scene:
    init_state : str # (s_0, a_0, s_1, s_2, ... ) => we need s_0 for start of a trajectory
    events : List[Event]
    #initial_val : Optional[float]

    @classmethod
    def from_json(cls, scene_json):
        # scene_json : list of dictionary
        init_state = 'Start recommendation.'
        events = []
        for i in range(len(scene_json)):
            events.append(AttributeEvent(scene_json[i]['attribute'], None, None, None))  
            events.append(ReviewEvent(scene_json[i]['review'], None, None, None))
        scene = cls(init_state, events)
        for p, n in zip(events[:-1], events[1:]):
            p.next = n
            n.prev= p
        for ev in events:
            ev.scene = scene
        return scene 

class RecommendData:
    def __init__(self, data_path: str):
        with open(data_path, 'r') as f:
            data= json.load(f)
        self.scenes = [Scene.from_json(data[i]) for i in range(len(data))]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        return self.scenes[i]

