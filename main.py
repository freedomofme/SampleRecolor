import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from image.scatter import createScatter

if __name__ == '__main__':
    createScatter('/Users/hhx/PycharmProjects/SampleRecolor/image/apple/000.png',
                  '/Users/hhx/PycharmProjects/SampleRecolor/image/apple/11.png')
