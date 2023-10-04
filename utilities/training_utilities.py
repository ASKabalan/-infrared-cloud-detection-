from jax import numpy as jnp
import time

class TrainingMetrics:
    def __init__(self):
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

    def reset(self):
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

    @property
    def avg_loss(self):
        return jnp.mean(jnp.stack(self.train_loss))

    @property
    def avg_accuracy(self):
        return jnp.mean(jnp.stack(self.train_accuracy))

    @property
    def avg_val_loss(self):
        return jnp.mean(jnp.stack(self.test_loss))

    @property
    def avg_val_acc(self):
        return jnp.mean(jnp.stack(self.test_accuracy))

class TrainingTimer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer hasn't been started yet!")
        elapsed_time = time.time() - self.start_time
        self.start_time = None
        return elapsed_time
