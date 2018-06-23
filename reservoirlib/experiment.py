

class BenchmarkExperiment:
    """
    Provides a framework for running an experiment using various tasks
    """

    def __init__(self, esn, task, num_training_trials):
        """
        :param esn: an object that can be trained on task data
            esn should conform to the BaseESN interface
        :param task: an object that will generate input and target time-series
            task should conform the BaseTask interface
        :param num_training_trials: number of trials to run the task
        """

        self.esn = esn
        self.task = task
        self.num_training_trials = num_training_trials

    def train_model(self):
        """
        Generates input and target signals and runs the ESN's training algorithm
        :return: None
        """
        # Generate data for training
        input_trials = [None for i in range(self.num_training_trials)]
        target_trials = [None for i in range(self.num_training_trials)]
        for i in range(self.num_training_trials):
            input_trials[i], target_trials[i] = self.task.generate_signal()

        # Train ESN
        self.esn.train(input_trials, target_trials) ######need train settings that task needs

    def evaluate_model(self):
        """
        Task specific validation runs
        :return: The task specific validation output
        """

        input_signal, target_output = self.task.generate_signal()
        prediction = self.esn.run(input_signal, output=True)
        return self.task.validate(prediction, target_output)
