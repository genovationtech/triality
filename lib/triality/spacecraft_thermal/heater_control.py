"""Heater control logic for spacecraft thermal management."""
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ControlMode(Enum):
    BANG_BANG = "bang_bang"
    PID = "pid"
    OFF = "off"

@dataclass
class PIDController:
    Kp: float = 1.0
    Ki: float = 0.1
    Kd: float = 0.01
    integral: float = 0.0
    prev_error: float = 0.0
    
    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        """PID control output"""
        error = setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return max(0, min(output, 1.0))  # Clamp [0,1]

@dataclass
class HeaterController:
    setpoint: float  # K
    deadband: float = 2.0  # K
    max_power: float = 100.0  # W
    mode: ControlMode = ControlMode.BANG_BANG
    pid: PIDController = None
    
    def __post_init__(self):
        if self.pid is None:
            self.pid = PIDController()
    
    def control_output(self, temperature: float, dt: float = 1.0) -> float:
        """Calculate heater power output [W]"""
        if self.mode == ControlMode.OFF:
            return 0.0
        elif self.mode == ControlMode.BANG_BANG:
            if temperature < self.setpoint - self.deadband:
                return self.max_power
            elif temperature > self.setpoint + self.deadband:
                return 0.0
            else:
                return self.max_power * 0.5  # Hysteresis zone
        else:  # PID
            duty_cycle = self.pid.update(self.setpoint, temperature, dt)
            return duty_cycle * self.max_power
