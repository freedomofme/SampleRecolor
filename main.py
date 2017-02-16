import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from image.scatter import readImage,createScatter

if __name__ == '__main__':

    createScatter('apple')

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # # Hold activation for multiple lines on same graph
    # plt.hold('on')
    # # Set x-axis range
    # plt.xlim((1,9))
    # # Set y-axis range
    # plt.ylim((1,9))
    # # Draw lines to split quadrants
    # plt.plot([4,6],[1,9], linewidth=4, color='red' )
    # # plt.plot([1,9],[5,5], linewidth=4, color='red' )
    # plt.title('Quadrant plot')
    # # Draw some sub-regions in upper left quadrant
    # plt.plot([3,3],[5,9], linewidth=2, color='blue')
    # plt.plot([1,5],[7,7], linewidth=2, color='blue')
    # plt.show()