# Built-in imports:
import sys

class Progress(object):
    def __init__(self, bar_width, number_of_steps):
        self.width = int(bar_width)         # width of progress bar in chars
        self.n = number_of_steps * 1.0      # total number of steps
        self.step = 0.0                     # current step

    def get_length(self):
        # Returns:
        #   string of characters to print for bar
        done = int(self.step/self.n*self.width)
        progress = "=" * done
        undone =   " " * (self.width - done)
        return progress + undone

    def start_progress(self):
        # Creates an empty progress bar and prints to screen
        sys.stdout.write('[' +  self.get_length() + ']')
        sys.stdout.flush()
        return

    def update_progress(self):
        # Updates progress bar and prints update to screen
        self.step += 1.0
        sys.stdout.write('\r[' +  self.get_length() + ']')
        sys.stdout.flush()
        return

    def complete_progress(self):
        # Makes sure the next line overrides the progress bar
        sys.stdout.write('\r')
