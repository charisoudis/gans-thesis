from PIL import Image
import h5py
import numpy as np
import utils.command_line_logger as cll


def main():
    logger = cll.CommandLineLogger(log_level='debug')
    logger.info('Fashion Synthesis Benchmark')

    hdf5_file_path = '/data/Datasets/DeepFashion/Fashion Synthesis Benchmark/Img/G2.h5'
    hdf5_file = h5py.File(hdf5_file_path, 'r')

    logger.info(str(list(hdf5_file.keys())))
    for key in list(hdf5_file.keys()):
        logger.info(f'{key} contains {len(hdf5_file[key])} images')

    # logger.info(hdf5_file['ih'][0].shape)

    image = hdf5_file['b_'][1001]
    image = np.reshape(image / image.max(), (128, 128))
    # image = np.reshape(image, (128, 128, 3))
    print(image.shape)
    img = Image.fromarray(image)
    # img.save('my.png')
    img.show()

    hdf5_file.close()


if __name__ == '__main__':
    main()
