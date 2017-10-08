import visdom
import numpy as np

viz = visdom.Visdom()

win = viz.line(
    X=np.column_stack((np.arange(0, 10), np.arange(0, 10))),
    Y=np.column_stack((np.linspace(5, 10, 10), np.linspace(5, 10, 10) + 5)),
)
viz.line(
    X=np.column_stack((np.arange(10, 20), np.arange(10, 20))),
    Y=np.column_stack((np.linspace(5, 10, 10), np.linspace(5, 10, 10) + 5)),
    win=win,
    update='append'
)
viz.updateTrace(
    X=np.arange(21, 30),
    Y=np.arange(1, 10),
    win=win,
    name='2'
)
viz.updateTrace(
    X=np.arange(31, 40),
    Y=np.arange(11, 20),
    win=win,
    name='2',
    append=True
)
