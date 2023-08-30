from lightning.pytorch import Callback
class BestPerformance(Callback):

    def __init__(self, monitor, mode):
        super().__init__()

        self.monitor = monitor
        assert monitor.split('_')[0] == 'dev'
        self.test_monitor = '_'.join(['test'] + monitor.split('_')[1:])

        self.mode = mode
        assert mode in ['max', 'min']
