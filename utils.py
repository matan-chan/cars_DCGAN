from os.path import isfile, join, isdir
from os import listdir
from pathlib import Path
import numpy as np
import threading
import zipfile
import cv2
import os


class DataPipe:
    """
    This class handling the data.

    Attributes
    ----------

    all_images : list
        list of all the images

    """

    def __init__(self):
        self.all_images = []

    def agent_standardize(self, start, end, all_images):
        """
        thread function which call the standardize_images function
        :param start:the start index of the thread`s section
        :type start: int
        :param end:the end index of the thread`s section
        :type end: int
        :param all_images:list of images full paths
        :type all_images: list
        """
        for i in range(start, end):
            cv2.imwrite(r"data" + "\\" + str(start + i) + '.jpg', self.standardize_images(cv2.imread(all_images[i])))

    @staticmethod
    def agent_loader(start, end, all_images, result, index):
        """
        thread function which load images
        :param start:the start index of the thread`s section
        :type start: int
        :param end:the end index of the thread`s section
        :type end: int
        :param all_images:list of images full paths
        :type all_images: list
        :param result:list of images
        :type result: list
        :param index:the thread`s` number
        :type index: int
        """
        arr = []
        for i in range(start, end):
            arr.append(cv2.imread(r"data" + "\\" + all_images[i]))
        arr = np.array(arr)
        arr = (arr.astype(np.float32) - 127.5) / 127.5
        result[index] = list(arr)

    @staticmethod
    def standardize_images(image):
        """
        standardize the images size to 180x100
        """
        height, width, _ = image.shape
        scale = 100 / height
        width = int(width * scale)
        height = int(height * scale)
        dim = (width, height) if width < 1.8 * height else (int(height * 1.8), height)
        resized_image = cv2.resize(image, dim)
        left = (180 - resized_image.shape[1]) // 2
        right = 180 - resized_image.shape[1] - left
        bottom = 100 - resized_image.shape[0]
        return cv2.copyMakeBorder(resized_image, 0, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    def load_all_images(self):
        """
        loads all the training images
        """
        threads = [None] * os.cpu_count()
        results = [None] * os.cpu_count()
        # if the is`nt "data" directory and there is "data.zip" this section will extract it
        if not isdir('data'):
            if isfile('data.zip'):
                with zipfile.ZipFile('data.zip', 'r') as zip_ref:
                    zip_ref.extractall('')
            else:
                print('no "data" directory')
                return np.array([])

        all_images = [f for f in listdir('data') if isfile(join('data', f))]
        part = len(all_images) // os.cpu_count()
        for i in range(len(threads)):
            too = len(all_images) if i + 1 == len(threads) else part * (i + 1)
            threads[i] = threading.Thread(target=self.agent_loader, args=(part * i, too, all_images, results, i))
            threads[i].start()
        for i in range(len(threads)):
            threads[i].join()
        final = []
        for i in range(len(threads)):
            final = final + results[i]
        self.all_images = final

    def get_butch(self, butch_size):
        """
        gets a butch of training images can be called right out *slow but ram efficient* or use "load_all_images"
        before and it will be *quick but ram hungry*
        """
        if self.all_images:
            images = []
            idx = np.random.randint(0, len(self.all_images), butch_size)
            for i in idx:
                images.append(self.all_images[i])
            return np.array(images)
        # if the is`nt "data" directory and there is "data.zip" this section will extract it
        if not isdir('data'):
            if isfile('data.zip'):
                with zipfile.ZipFile('data.zip', 'r') as zip_ref:
                    zip_ref.extractall('')
            else:
                print('no "data" directory')
                return np.array([])

        threads = [None] * os.cpu_count()
        results = [None] * os.cpu_count()
        all_images = [f for f in listdir('data') if isfile(join('data', f))]
        idx = np.random.randint(0, len(all_images), butch_size)
        part_of_images = []
        for i in idx:
            part_of_images.append(all_images[i])
        part = len(part_of_images) // os.cpu_count()
        for i in range(len(threads)):
            too = len(part_of_images) if i + 1 == len(threads) else part * (i + 1)
            threads[i] = threading.Thread(target=self.agent_loader, args=(part * i, too, part_of_images, results, i))
            threads[i].start()
        for i in range(len(threads)):
            threads[i].join()
        final = []
        for i in range(len(threads)):
            final = final + results[i]
        return np.array(final)

    def import_data(self, paths):
        """
        import all the images to our local "data" folder and standardize them
        :param paths:all the folder`s paths from which we would like to draw our`s data
        :type paths: list
        """
        threads = [None] * os.cpu_count()
        all_images = []
        for path in paths:
            if '.zip' == Path(path).suffix:
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(Path(path).parent.absolute())
                    path = path[:len(path) - 4]

            all_images += [path + "\\" + f for f in listdir(path) if isfile(join(path, f))]
        part = len(all_images) // os.cpu_count()
        for i in range(len(threads)):
            if i + 1 == len(threads):
                threads[i] = threading.Thread(target=self.agent_standardize,
                                              args=(part * i, len(all_images), all_images))
            else:
                threads[i] = threading.Thread(target=self.agent_standardize,
                                              args=(part * i, part * (i + 1), all_images))
            threads[i].start()
