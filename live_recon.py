import matplotlib.pyplot as plt
from bluesky import RunEngine
from bluesky.examples import Reader, Mover
from bluesky.plans import scan
from bluesky.callbacks import LiveTable, CallbackBase
from bluesky.utils import install_qt_kicker
import tomopy
import numpy as np

L = 64
obj = tomopy.lena(L)

det = Reader('det', {'image': lambda: tomopy.project(obj, angle.read()['angle']['value'])})
angle = Mover('angle', {'angle': lambda x: x}, {'x': 0})
angle._fake_sleep = 0.01


RE = RunEngine({})
install_qt_kicker()
t = LiveTable([angle])


class LiveRecon(CallbackBase):
    SMALL = 1e-6

    def __init__(self, name, x, y, window_size, ax=None, **recon_kwargs):
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        ax.set_title('Reconstruction using Tomopy')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        self.im = ax.imshow(np.zeros((y, x)), origin='upper')
        recon_kwargs.setdefault('num_gridx', x)
        recon_kwargs.setdefault('num_gridy', y)
        self._name = name
        self._x, self._y = x, y
        self._recon_kwargs = recon_kwargs
        self._window_size = window_size
        self._data_ring = [None] * window_size
        self._angle_ring = np.zeros(window_size)

    def start(self, doc):
        self._partial = self.SMALL * np.ones((1, self._y, self._x))

    def event(self, doc):
        data = doc['data'][self._name]
        angle = doc['data']['angle']
        seq_num = doc['seq_num'] - 1
        self._data_ring[seq_num % self._window_size] = data
        data_ring = np.expand_dims(self._data_ring, axis=1)
        self._angle_ring[seq_num % self._window_size] = angle
        self._partial = tomopy.recon(np.asarray(data_ring[:seq_num + 1]),
                                     self._angle_ring[:seq_num + 1],
                                     **self._recon_kwargs,
                                     init_recon=self._partial)
        self.im.set_data(self._partial)
        self.im.set_clim((np.min(self._partial), np.max(self._partial)))
        self.im.figure.canvas.draw_idle()


class LiveSinogram(CallbackBase):
    def __init__(self, name, width, ax=None):
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        ax.set_title('Sinogram')
        ax.set_xlabel('sequence number')
        ax.set_ylabel('detector position')
        self.im = ax.imshow(np.zeros((1, width)), aspect='auto')
        ax.figure.colorbar(self.im, ax=ax)
        self._name = name
        self._width = width

    def start(self, doc):
        self._cache = []

    def event(self, doc):
        self._cache.append(doc['data'][self._name][0][0])
        arr = np.asarray(self._cache)
        self.im.set_data(arr.T)
        self.im.set_extent((0, len(self._cache), 0, self._width))
        self.im.set_clim((arr.min(), arr.max()))
        self.im.figure.canvas.draw_idle()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
p = LiveSinogram('image', 94, ax=ax1)
r = LiveRecon('image', L, L, window_size=3, algorithm='sirt', ax=ax2)

RE(scan([det], angle, 0, np.pi, 100), [t, p, r])
