from jax import numpy as jnp
import time

class TrainingMetrics:
    """
    A class to track and compute average training and testing metrics during model training.
    """
    def __init__(self):
        """
        Initializes lists to store training and testing loss and accuracy values.
        """
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

    def reset(self):
        """
        Resets all the lists storing training and testing metrics.
        """
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

    @property
    def avg_loss(self):
        """
        Computes the average training loss from the stored values.
        
        Returns:
        - float: Average training loss.
        """
        return jnp.mean(jnp.stack(self.train_loss))

    @property
    def avg_accuracy(self):
        """
        Computes the average training accuracy from the stored values.
        
        Returns:
        - float: Average training accuracy.
        """
        return jnp.mean(jnp.stack(self.train_accuracy))

    @property
    def avg_val_loss(self):
        """
        Computes the average testing/validation loss from the stored values.
        
        Returns:
        - float: Average testing/validation loss.
        """
        return jnp.mean(jnp.stack(self.test_loss))

    @property
    def avg_val_acc(self):
        """
        Computes the average testing/validation accuracy from the stored values.
        
        Returns:
        - float: Average testing/validation accuracy.
        """
        return jnp.mean(jnp.stack(self.test_accuracy))

class TrainingTimer:
    """
    A simple timer class to measure elapsed time.
    """
    def __init__(self):
        """
        Initializes the timer with a None start time.
        """
        self.start_time = None

    def start(self):
        """
        Starts the timer by recording the current time.
        """
        self.start_time = time.time()

    def stop(self):
        """
        Stops the timer and computes the elapsed time since it was started.
        
        Returns:
        - float: Elapsed time in seconds.
        
        Raises:
        - ValueError: If the timer was not started before calling this method.
        """
        if self.start_time is None:
            raise ValueError("Timer hasn't been started yet!")
        elapsed_time = time.time() - self.start_time
        self.start_time = None
        return elapsed_time
