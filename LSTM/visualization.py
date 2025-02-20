from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    """
    A wrapper for TensorBoard's SummaryWriter to simplify logging metrics.

    Attributes
    ----------
    log_dir : str
        Directory to save TensorBoard logs.
    writer : SummaryWriter
        Instance of TensorBoard's SummaryWriter.
    """
    def __init__(self, log_dir: str = "tensorboard_logs"):
        """
        Initializes the TensorBoard logger.

        Parameters
        ----------
        log_dir : str, optional
            Directory to save TensorBoard logs. Default is "tensorboard_logs".
        """
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        """
        Logs a scalar value to TensorBoard.

        Parameters
        ----------
        tag : str
            The name of the metric (e.g., "loss", "accuracy").
        value : float
            The value of the metric.
        step : int
            The current training step or epoch.
        """
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values, step: int):
        """
        Logs a histogram of a tensor's values to TensorBoard.

        Parameters
        ----------
        tag : str
            The name of the tensor.
        values : torch.Tensor
            The tensor to log.
        step : int
            The current training step or epoch.
        """
        self.writer.add_histogram(tag, values, step)

    def close(self):
        """Closes the TensorBoard writer."""
        self.writer.close()

