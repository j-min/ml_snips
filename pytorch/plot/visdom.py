from visdom import Visdom
import numpy as np


class VisdomWriter(object):
    def __init__(self, env=None):
        """Extended Visdom Writer"""
        self.vis = Visdom(env=env)
        
    def update_text(self, text):
        """Text Memo (usually used to note hyperparameter-configurations)"""
        self.vis.text(text)


    def update_loss(self, step_i, loss, title='Learning Curve', xlabel='Epoch', ylabel='Loss'):
        """Update loss (X: Step (Epoch) / Y: loss)"""

        if step_i == 1:
            # line plot
            self.win = self.vis.line(
                X=np.array([step_i]),
                Y=np.array([loss]),
                opts=dict(
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                ))

        else:
            self.vis.updateTrace(
                X=np.array([step_i]),
                Y=np.array([loss]),
                win=self.win)
