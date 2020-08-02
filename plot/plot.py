import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyaudio
from scipy.fftpack import fft

np.seterr(divide='ignore')  # for zero division in np.log10(0)

chunk = 1024
rate = 44100
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)

fig, (ax1, ax2) = plt.subplots(2)

t = np.arange(0, chunk)
a1, = ax1.plot([0], [0])

ax1.set_xlim(0, chunk)
ax1.set_ylim(-1, 1)
ax1.grid()

f = np.linspace(0, rate, chunk)
a2, = ax2.plot([0], [0])

ax2.set_xlim(rate / chunk, rate / 2)
ax2.set_ylim(-80, 0)
ax2.grid()


def animate(i):
    x = np.frombuffer(stream.read(chunk, exception_on_overflow=False), dtype="int16") / 32768
    a1.set_data(t, x)
    a2.set_data(f, 20 * np.log10(np.abs(fft(x)) / chunk))
    return a1, a2,


ani = animation.FuncAnimation(fig, animate, interval=10, blit=True)

plt.show()
