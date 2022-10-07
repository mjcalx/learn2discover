from abc import abstractmethod
from loggers.logger_factory import LoggerFactory
from utils.logging_utils import Verbosity
from typing import List

class AbstractSubject:
    @abstractmethod
    def attach(self, observer: 'AbstractObserver') -> None: pass
    @abstractmethod
    def detach(self, observer: 'AbstractObserver') -> None: pass
    @abstractmethod
    def notify(self) -> None: pass

class AbstractObserver:
    @abstractmethod
    def update(self, subject: AbstractSubject) -> None: pass

class Observer(AbstractObserver):
    def __init__(self, **kw):
        super(Observer, self).__init__(**kw)

class Subject(AbstractSubject):
    _observers: List[Observer] = []

    def __init__(self, **kw):
        super(Subject, self).__init__(**kw)

    @property
    def observers(self):
        return self._observers
    
    def attach(self, observer: Observer) -> None:
        self.logger.debug(f'Attaching observer "{observer.__class__.__name__}"', verbosity=Verbosity.TALKATIVE)
        self.observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        self.logger.debug(f'Detaching observer "{observer.__class__.__name__}"', verbosity=Verbosity.TALKATIVE)
        self.observers.remove(observer)
    
    def notify(self) -> None:
        self.logger.debug(f'Notifying observers...', verbosity=Verbosity.TALKATIVE)
        for observer in self._observers:
            observer.update(self)