import sys

class Progress(object):
    def __init__(self, bar_width, number_of_steps):
        self.width = int(bar_width)
        self.n = number_of_steps * 1.0
        self.step = 0.0

    def get_length(self):
        done = int(self.step/self.n*self.width)
        progress = "=" * done
        undone =   " " * (self.width - done)
        return progress, undone

    def start_progress(self):
        f, e = self.get_length()
        sys.stdout.write('[' + f + e + ']')
        sys.stdout.flush()
        return

    def update_progress(self):
        self.step += 1.0
        f, e = self.get_length()
        sys.stdout.write('\r[' + f + e + ']')
        sys.stdout.flush()
        return

    def complete_progress(self):
        sys.stdout.write('\r')
