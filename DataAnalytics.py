import matplotlib.pyplot as plt

class DataAnalytics:
    def __init__(self):
        self.drawChart()

    def drawChart(self):
        time = [0, 1, 2, 3]
        position = [0, 100, 200, 300]

        plt.plot(time, position)
        plt.xlabel('Time (hr)')
        plt.ylabel('Position (km)')
        plt.show()