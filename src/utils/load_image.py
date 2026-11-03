import tensorflow as tf
from src.config import IMG_SIZE


def load_image(path, label):
    """
    Đọc ảnh từ đường dẫn, giải mã ảnh RGB và chuẩn bị nhãn tương ứng.

    Parameters:
        path (tf.Tensor): Đường dẫn đến file ảnh.
        label (tf.Tensor): Nhãn số nguyên của ảnh.

    Returns:
        tuple: Ảnh đã đọc và nhãn tương ứng.
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_image(
        image,
        channels = 3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label