"""Geometric domains for PDEs"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Interval:
    """1D interval [a, b]"""
    a: float
    b: float
    dim: int = 1

    def __post_init__(self):
        if self.a >= self.b:
            raise ValueError(f"Invalid interval: {self.a} >= {self.b}")

    def length(self): return self.b - self.a
    def __repr__(self): return f"[{self.a}, {self.b}]"


@dataclass
class Rectangle:
    """2D rectangle [xmin, xmax] × [ymin, ymax]"""
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    dim: int = 2

    def __post_init__(self):
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError("Invalid rectangle dimensions")

    def area(self): return (self.xmax - self.xmin) * (self.ymax - self.ymin)
    def __repr__(self): return f"[{self.xmin}, {self.xmax}] × [{self.ymin}, {self.ymax}]"


@dataclass
class Square:
    """2D square [0, L]²"""
    L: float
    dim: int = 2

    def area(self): return self.L ** 2
    def __repr__(self): return f"[0, {self.L}]²"


@dataclass
class Circle:
    """2D circle with center and radius"""
    center: tuple
    radius: float
    dim: int = 2

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("Radius must be positive")

    def area(self): return np.pi * self.radius ** 2
    def __repr__(self): return f"Circle(r={self.radius})"
