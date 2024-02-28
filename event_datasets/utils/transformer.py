import numpy as np
import random
from . import event as ev
import math

__all__ = ["EventReorder", "EventReverse", "EventDrift"]


'''combined strategy'''
class EventReorder:
    def __init__(self, p='p', x='x', y='y', t='t') -> None:
        permit = ['p', 'x', 'y', 't']
        assert p in permit and x in permit and y in permit and t in permit, f"args can only in {permit}."
        self.p, self.x, self.y, self.t = p, x, y, t

    def __call__(self, evdata:ev.event):
        return ev.event(evdata[self.p], evdata[self.x], evdata[self.y], evdata[self.t])
    
    def __repr__(self):
        return self.__class__.__name__ + f'\treorder: {self.p}, {self.x}, {self.y}, {self.t}'


'''
reverse space(x, y, xy-yx), temporal(t, p)
random (yes or not) (super range) (random choice)
range 比例的内部实验。
'''

class EventReverse:
    identical = 0
    randomReverse = 1
    xRange = 2
    yRange = 3
    xyRange= 4
    pRange = 5
    xAll = 6 # useful
    yAll = 7 # useless
    pAll = 8
    tALL = 9
    tRange = 10
    def __init__(self, HW, methods, probabilities:list=None, number=1, uniform=False):
        '''
        d are ['t', 'x', 'y', 'p', 'xy']
        events filter are "r-random" "rr-range ratio" "a-all"
        '''
        number = min(len(methods), number)
        self.number = number
        self.p = probabilities
        self.HW = HW
        self.uniform = uniform
        self.method_dict = {EventReverse.identical:['i',(), 0],
            EventReverse.randomReverse:['r',(), 0.3],
            EventReverse.xRange: ['rr', ('x'), 0.5],
            EventReverse.yRange: ['rr', ('y'), 0.5],
            EventReverse.xyRange: ['rr', ('x', 'y'), 0.5],
            EventReverse.pRange: ['rr', ('p'), 0.5],
            EventReverse.xAll: ['a', ('x'), 1],
            EventReverse.yAll: ['a', ('y'), 1],
            EventReverse.tALL: ['a', ('t'), 1],
            EventReverse.pAll: ['a', ('p'), 1],
            EventReverse.tRange: ['rr', ('t'), 0.5]
            }
        if isinstance(methods, list) or isinstance(methods, tuple):
            self.methods = methods
        elif isinstance(methods, dict):
            self.methods = list(methods.keys())
            for m in self.methods:
                self.method_dict[m][2] = methods[m]

    def analyse(self, methods):
        d_events = []
        for m in methods:
            d_events.append(self.method_dict[m])
        return d_events

    def __call__(self, evdata:ev.event):
        method = list(np.random.choice(self.methods, self.number, replace=False, p=self.p))
        d_event = self.analyse(method)
        for elem in d_event:
            if elem[0] == 'i':
                return evdata
                # pass
            elif elem[0] == 'r':
                if self.uniform:
                    elem[2] = np.random.uniform(0, elem[2])
                domains = np.random.choice(('x', 'y', 't', 'p'), np.random.randint(4),replace=False)
                select_events = np.random.choice(evdata.length, int(evdata.length*elem[2]),replace=False)               
                for d in domains:
                    evdata.domainReverse(d, self.HW, select_events)
            elif elem[0] == 'rr':
                domains = elem[1]
                if self.uniform:
                    elem[2] = np.random.uniform(0, elem[2])
                start = np.random.randint(int(evdata.length*(1-elem[2]))+1)
                select_events = slice(start, start+int(evdata.length*elem[2]))  
                for d in domains:
                    evdata.domainReverse(d, self.HW, select_events)
            else: # a
                domains = elem[1]
                select_events = slice(evdata.length)
                for d in domains:
                    evdata.domainReverse(d, self.HW, select_events)
        return evdata

    def __repr__(self):
        return self.__class__.__name__ + f'\ttransformer times:{self.number}\tHW:{self.HW}\
            \nmethods:{self.analyse(self.methods)}\nprobabilities:{self.p}\nuniform:{self.uniform}'

'''
erase + drop: drift(/reverse event/no trigger) out, drift specific, 
drift recurrently / parallelly
drift space:(x, y, x+y), temporal (t)
random (yes or not)(super range) (random choice)
'''    
class EventDrift:
    identical = 0
    randomDrift = 1
    xRange = 2
    yRange = 3
    xyRange= 4
    pRange = 5
    xAll = 6
    yAll = 7
    pAll = 8
    tALL = 9
    xyALL = 10
    xytALL = 11
    # tRange = 11
    # xyFiled = 12
    # current = 11
    # distance = 0.1
    def __init__(self, HW, methods, probabilities=None, number=1, uniform=False):
        number = min(len(methods), number)
        self.number = number
        self.p = probabilities
        self.HW = HW
        self.uniform = uniform
        self.method_dict = {EventDrift.identical:['i',(), 0,0,False],
            EventDrift.randomDrift:['r',(), 0.3, 0.2, False],
            EventDrift.xRange: ['rr', ('x'), 0.5, 0.15, False],
            EventDrift.yRange: ['rr', ('y'), 0.5, 0.15, False],
            EventDrift.xyRange: ['rr', ('x', 'y'), 0.5, 0.15, False],
            EventDrift.pRange: ['rr', ('p'), 0.5, 0.15, False],
            EventDrift.xAll: ['a', ('x'), 1, 0.15, False],
            EventDrift.yAll: ['a', ('y'), 1, 0.15, False],
            EventDrift.tALL: ['a', ('t'), 1, 0.15, False],
            EventDrift.pAll: ['a', ('p'), 1, 0.15, False],
            EventDrift.xyALL: ['a', ('x', 'y'), 1, 0.15, False],
            EventDrift.xytALL: ['a', ('x', 'y', 't'), 1, 0.15, False],
            }
        if isinstance(methods, list) or isinstance(methods, tuple):
            self.methods = methods
        elif isinstance(methods, dict):
            self.methods = list(methods.keys())
            for m in self.methods:
                self.method_dict[m][2:] = methods[m]

    def analyse(self, methods):
        d_events = []
        for m in methods:
            d_events.append(self.method_dict[m])
        return d_events

    def __call__(self, evdata:ev.event):
        method = list(np.random.choice(self.methods, self.number, replace=False, p=self.p))
        d_event = self.analyse(method)
        for elem in d_event:
            if elem[0] == 'i':
                return evdata
                # pass
            elif elem[0] == 'r':
                if self.uniform:
                    elem[2] = np.random.uniform(0, elem[2])
                domains = np.random.choice(('x', 'y', 't', 'p'), np.random.randint(4),replace=False)
                select_events = np.random.choice(evdata.length, int(evdata.length*elem[2]),replace=False)               
                for d in domains:
                    evdata.domainDrift(d, self.HW, select_events, elem[3], elem[4])
            elif elem[0] == 'rr':
                domains = elem[1]
                if self.uniform:
                    elem[2] = np.random.uniform(0, elem[2])
                start = np.random.randint(int(evdata.length*(1-elem[2]))+1)
                select_events = slice(start, start+int(evdata.length*elem[2]))  
                for d in domains:
                    evdata.domainDrift(d, self.HW, select_events, elem[3], elem[4])
            else: # a
                domains = elem[1]
                select_events = slice(evdata.length)
                for d in domains:
                    evdata.domainDrift(d, self.HW, select_events, elem[3], elem[4])
        return evdata

    def __repr__(self):
        return self.__class__.__name__ + f'\ttransformer times:{self.number}\tHW:{self.HW}\
            \nmethods:{self.analyse(self.methods)}\nprobabilities:{self.p}'

class EventDrop:
    def __init__(self, HW):
        self.HW = HW

    def __call__(self, evdata:ev.event):
        evdata.event_drop(self.HW)
        return evdata
    
    def __repr__(self):
        return self.__class__.__name__ + f'\tHW:{self.HW}'

class EventRotation:
    identical = 0
    randomRotation1 = 1
    randomRotation2 = 2
    spaceRotation= 3
    xtimeRotation = 4
    ytimeRotation = 5
    timeRotation = 6
    realxtimeRotation = 7
    realytimeRotation = 8
    realtimeRotation = 9
    mixspaceRotation = 10
    randomRotation = 11
    XTRotation = 12
    YTRotation = 13
    XYRotation = 14
    def __init__(self, HW, methods, drop=False, setscaling=True, probabilities=None, number=1, uniform=False):
        number = min(len(methods), number)
        self.number = number
        self.p = probabilities
        self.HW = HW
        self.uniform = uniform
        self.drop = drop
        self.setscaling = setscaling
        self.method_dict = {EventRotation.identical:['i',[0,0], [0.5,0.5]],
            EventRotation.randomRotation1:['rr1',[-math.pi,math.pi], [0.5,0.5]],
            EventRotation.randomRotation2:['rr2',[-math.pi,math.pi], [0.5,0.5]],
            EventRotation.spaceRotation: ['sr',[math.pi/2,math.pi/2], [0.5,0.5]],
            EventRotation.xtimeRotation: ['xtr',[math.pi/2,math.pi/2], [0.5,0.5]],
            EventRotation.ytimeRotation: ['ytr',[math.pi/2,math.pi/2], [0.5,0.5]],
            EventRotation.timeRotation: ['tr',[math.pi/2,math.pi/2], [0.5,0.5]],
            EventRotation.realxtimeRotation: ['realxtr',[math.pi/2,math.pi/2], [0.5,0.5]],
            EventRotation.realytimeRotation: ['realytr',[math.pi/2,math.pi/2], [0.5,0.5]],
            EventRotation.realtimeRotation: ['realtr',[math.pi/2,math.pi/2], [0.5,0.5]],
            EventRotation.mixspaceRotation: ['mixsr',[math.pi/2,math.pi/2], [0.5,0.5]],
            EventRotation.randomRotation: ['rr',[math.pi/2,math.pi/2], [0.5,0.5]],
            EventRotation.XYRotation: ['xy',[math.pi/2,math.pi/2], [0.5,0.5]],
            EventRotation.XTRotation: ['xt',[math.pi/2,math.pi/2], [0.5,0.5]],
            EventRotation.YTRotation: ['yt',[math.pi/2,math.pi/2], [0.5,0.5]],
            }
        if isinstance(methods, list) or isinstance(methods, tuple):
            self.methods = methods
        elif isinstance(methods, dict):
            self.methods = list(methods.keys())
            for m in self.methods:
                self.method_dict[m][1:] = methods[m]

    def analyse(self, methods):
        d_events = []
        for m in methods:
            d_events.append(self.method_dict[m])
        return d_events

    def __call__(self, evdata:ev.event):
        self.methods.append(0)
        method = list(np.random.choice(self.methods, self.number, replace=False, p=self.p))
        d_event = self.analyse(method)
        for elem in d_event:
            if elem[0] == 'i':
                return evdata
            elif elem[0] == 'xy':
                domains = list(np.random.choice(['x','y'], 2, replace=False, p=self.p))
                angle = np.random.uniform(elem[1][0], elem[1][1])
                centers = {domains[0]:np.random.uniform(elem[2][0], elem[2][1]), domains[1]:np.random.uniform(elem[2][0], elem[2][1])}
                evdata.event_rotation2(domains, self.HW, angle, centers, setscaling=self.setscaling, drop=self.drop)
            elif elem[0] == 'xt':
                domains = list(np.random.choice(['x','t'], 2, replace=False, p=self.p))
                angle = np.random.uniform(elem[1][0], elem[1][1])
                centers = {domains[0]:np.random.uniform(elem[2][0], elem[2][1]), domains[1]:np.random.uniform(elem[2][0], elem[2][1])}
                evdata.event_rotation2(domains, self.HW, angle, centers, setscaling=self.setscaling, drop=self.drop)
            elif elem[0] == 'yt':
                domains = list(np.random.choice(['y','t'], 2, replace=False, p=self.p))
                angle = np.random.uniform(elem[1][0], elem[1][1])
                centers = {domains[0]:np.random.uniform(elem[2][0], elem[2][1]), domains[1]:np.random.uniform(elem[2][0], elem[2][1])}
                evdata.event_rotation2(domains, self.HW, angle, centers, setscaling=self.setscaling, drop=self.drop)
            elif elem[0] == 'rr':
                domains = list(np.random.choice(['x','y','t'], 2, replace=False, p=self.p))
                angle = np.random.uniform(elem[1][0], elem[1][1])
                centers = {domains[0]:np.random.uniform(elem[2][0], elem[2][1]), domains[1]:np.random.uniform(elem[2][0], elem[2][1])}
                evdata.event_rotation2(domains, self.HW, angle, centers, setscaling=self.setscaling, drop=self.drop)
            elif elem[0] == 'sr':
                angle = np.random.uniform(elem[1][0], elem[1][1])
                center = np.random.uniform(elem[2][0], elem[2][1])
                evdata.event_rotation('s', self.HW, angle, center)
            elif elem[0] == 'xtr':
                angle = np.random.uniform(elem[1][0], elem[1][1])
                center = np.random.uniform(elem[2][0], elem[2][1])
                evdata.event_rotation('xt', self.HW, angle, center, self.drop)
            elif elem[0] == 'ytr':
                angle = np.random.uniform(elem[1][0], elem[1][1])
                center = np.random.uniform(elem[2][0], elem[2][1])
                evdata.event_rotation('yt', self.HW, angle, center, self.drop)
            elif elem[0] == 'tr':
                angle = [np.random.uniform(elem[1][0], elem[1][1]), np.random.uniform(elem[1][0], elem[1][1])]
                center = np.random.uniform(elem[2][0], elem[2][1])
                evdata.event_rotation('t', self.HW, angle, center, self.drop)
            elif elem[0] == 'realxtr':
                angle = np.random.uniform(elem[1][0], elem[1][1])
                center = np.random.uniform(elem[2][0], elem[2][1])
                evdata.event_rotation('realxt', self.HW, angle, center, self.drop)
            elif elem[0] == 'realytr':
                angle = np.random.uniform(elem[1][0], elem[1][1])
                center = np.random.uniform(elem[2][0], elem[2][1])
                evdata.event_rotation('realyt', self.HW, angle, center, self.drop)
            elif elem[0] == 'realtr':
                angle = [np.random.uniform(elem[1][0], elem[1][1]), np.random.uniform(elem[1][0], elem[1][1])]
                center = np.random.uniform(elem[2][0], elem[2][1])
                evdata.event_rotation('realt', self.HW, angle, center, self.drop)
            elif elem[0] == 'rr1':
                domains = np.random.choice(('s', 'xt', 'yt'), np.random.randint(3),replace=False)
                angle = np.random.uniform(elem[1][0], elem[1][1])
                center = np.random.uniform(elem[2][0], elem[2][1])
                for d in domains:
                    evdata.event_rotation(d, self.HW, angle, center)
            elif elem[0] == 'rr2':
                domains = np.random.choice(('s', 'xt', 'yt', 'realyt', 'realxt'), np.random.randint(5),replace=False)
                angle = np.random.uniform(elem[1][0], elem[1][1])
                center = np.random.uniform(elem[2][0], elem[2][1])
                for d in domains:
                    evdata.event_rotation(d, self.HW, angle, center)
            elif elem[0] == 'mixsr':  
                angle = [np.random.uniform(elem[1][0], elem[1][1]), np.random.uniform(elem[1][0], elem[1][1])]
                center = np.random.uniform(elem[2][0], elem[2][1])
                evdata.event_rotation('mixs', self.HW, angle, center)     
        return evdata
    
    def __repr__(self):
        return self.__class__.__name__ + f'\ttransformer times:{self.number}\tHW:{self.HW}\
            \nmethods:{self.analyse(self.methods)}\ndrop:{self.drop}\nprobabilities:{self.p}'

class RandomCrop(object):
    """Crop the given DVS at a random location.
    input ndarray
    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (int(size), int(size))
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img : Image to be cropped CHWT.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[1], img.shape[2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def SNNpadding(img, border=None, fill=0):
        # CHWT
        if isinstance(border, tuple):
            if len(border) == 2:
                left, top = right, bottom = border
            elif len(border) == 4:
                left, top, right, bottom = border
        else:
            left = top = right = bottom = border
        return np.pad(img, ((0, 0), (top, bottom), (left, right), (0, 0)), mode='constant', constant_values=fill)

    def __call__(self, img):
        """
        Args:
            img (ndarray): CHWT to be cropped.

        Returns:
            img (ndarray): Cropped image.
        """
        if self.padding is not None:
            img = self.SNNpadding(img, self.padding, self.fill)

        i, j, h, w = self.get_params(img, self.size)

        return img[:, i:i+h, j:j+w, :].copy()

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class RandomTranspose(object):
    """Horizontally flip the given ndarray randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (ndarray): CHWT to be transpose.

        Returns:
            img (ndarray): transposed image.
        """
        if random.random() < self.p:
            return img.swapaxes(1, 2).copy()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomHorizontalFlip(object):
    """Horizontally flip the given ndarray randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (ndarray): Image to be flipped.

        Returns:
            ndarray: Randomly flipped image.
        """
        if random.random() < self.p:
            return np.flip(img, 2).copy()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)