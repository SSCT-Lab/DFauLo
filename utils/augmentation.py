# Author: Gentry Atkinson
# Organization: Texas University
# Data: 5 May, 2022
# Alter signals but keep them recognizable

from random import random, seed, gauss, randint, choice
import numpy as np


# Flip
# Invert the signal over the x (time) axis
def flip(originalSignal):
    return [-1 * x for x in originalSignal]


# Random Signal Drop
# Set some samples to 0 with random chance
def rand_signal_drop(originalSignal, chance=0.1):
    seed()
    return [i if random() > chance else 0 for i in originalSignal]


# Windowed Signal Drop
# Set a range of samples to 0
def wind_signal_drop(originalSignal, start=None, stop=None):
    seed()
    signalLength = len(originalSignal)
    if start == None and stop == None:
        start = randint(signalLength // 4, signalLength // 2 - 1)
        stop = start + int(gauss(0.1, 0.05) * signalLength)
        if stop >= signalLength: stop = signalLength - 1
    elif start == None:
        start = stop - int(gauss(0.1, 0.05) * signalLength)
        if start < 0: start = 0
    elif stop == None:
        stop = start + int(gauss(0.1, 0.05) * signalLength)
        if stop >= signalLength: stop = signalLength - 1
    return [0 if i >= start and i <= stop else originalSignal[i] for i in range(signalLength)]


# Fixed Signal Drop
# Set every nth sample to 0
def fixed_signal_drop(originalSignal, period=5):
    if period == None:
        seed()
        period = randint(5, 20)
    return [0 if i % period == 0 else originalSignal[i] for i in range(len(originalSignal))]


# Gaussian Noise Injection
# Add a small, random variance to each sample
def gau_noise_inj(originalSignal, sigma=None):
    seed()
    if sigma == None:
        sigma = 0.02 * (max(originalSignal) - min(originalSignal))
    return [gauss(i, sigma) for i in originalSignal]


# Amplitude Shift
# Add a fixed amount to the amplitude of the signal
def amp_shift(originalSignal, shift=None):
    if shift == None:
        seed()
        shift = choice([-0.1, 0.1]) * (max(originalSignal) - min(originalSignal))
    return [i + shift for i in originalSignal]


# Time Shift
# Move a signal forward or backwards in time
def time_shift(originalSignal, shift=None):
    signalLength = len(originalSignal)
    if shift == None:
        seed()
        shift = int(choice([-0.2, 0.2]) * signalLength)
    return [originalSignal[(i + shift) % signalLength] for i in range(signalLength)]