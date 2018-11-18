import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data_jpg = tf.gfile.FastGFile('image1.jpg', 'rb').read()
image_raw_data_png = tf.gfile.FastGFile('image1.png', 'rb').read()

with tf.Session() as sess:
	img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg) #图像解码
	img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8) #改变图像数据的类型

	img_data_png = tf.image.decode_png(image_raw_data_png)
	img_data_png = tf.image.convert_image_dtype(img_data_png, dtype=tf.uint8)

	plt.figure(1) #图像显示
	plt.imshow(img_data_jpg.eval())
	plt.figure(2)
	plt.imshow(img_data_png.eval())
	plt.show()

    
    
