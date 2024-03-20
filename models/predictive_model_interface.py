from abc import ABC, abstractmethod

class PredictiveModelInterface(ABC):
    @abstractmethod
    def predict(self, input_data):
        """
        Make a prediction on the input data.

        Parameters:
        - input_data: Input data for prediction.

        Returns:
        - Resulting prediction.
        """
        pass

    @abstractmethod
    def train(self, training_data, target):
        """
        Train the model with the provided training data and targets.

        Parameters:
        - training_data: Training data.
        - target: Targets corresponding to the training data.

        Returns:
        None
        """
        pass

    @abstractmethod
    def partial_fit(self, new_data, new_target):
        """
        Perform online training of the model using new data and targets.

        Parameters:
        - new_data: New data for online training.
        - new_target: Targets corresponding to the new data.

        Returns:
        None
        """
        pass