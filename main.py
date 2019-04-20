# coding: utf-8
#PedJointNet行人頭肩偵測系統主程序
import numpy as np
import os
import tensorflow as tf
import tkinter
from tkinter import *
from PIL import Image,ImageTk
import tkinter.filedialog
sys.path.append("..")
from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

class Applications():

    def load_image_into_numpy_array(self,image):
      (im_width, im_height) = image.size
      return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)

    def run_inference_for_single_image(self, image, graph):
      with graph.as_default():
        with tf.Session() as sess:
          # Get handles to input and output tensors
          ops = tf.get_default_graph().get_operations()
          all_tensor_names = {output.name for op in ops for output in op.outputs}
          tensor_dict = {}
          for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
          ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                  tensor_name)
          if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
          image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

          # Run inference
          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
      return output_dict

    def __init__(self):
        # Frame.__init__(self, master, bg='black')
        self.root = Tk()
        self.root.title('PedJointNet行人頭肩偵測系統')
        self.root.geometry('1280x1280')
        self.e = tkinter.StringVar()
        self.L1 = Label(self.root)
        self.lbA = Label(self.root, text='PedJointNet行人頭肩偵測系統', bg='#22C9C9', fg='white',
                         font=('微软雅黑', 36), width='34')
        self.lbA.grid(row=0, column=0, columnspan=10)
        # 创建两个按钮
        self.b1 = Button(self.root, text='輸入影像', bg='SlateGray', fg='white', font=('微软雅黑', 13, "bold"), width=34, height=4,
                    command=self.load_image)
        self.b1.grid(row=1, column=0, sticky=W, padx=80, pady=80)
        self.b2 = Button(self.root, text='一鍵偵測', bg='SkyBlue', fg='white', font=('微软雅黑', 13, "bold"), width=34, height=4,
                    command=self.detection)
        self.b2.grid(row=1, column=1, sticky=W, padx=80, pady=80)
        # self.selectFileName = tkinter.filedialog.askopenfilename(title='选择影像')

    def load_image(self):
        self.selectFileName = tkinter.filedialog.askopenfilename(title='选择影像')
        img_open = Image.open(self.selectFileName)
        self.img = img_open.resize((500, 500))
        self.img = ImageTk.PhotoImage(self.img)
        self.L1.image = self.img  # keep a reference
        Label(image=self.img).grid(row=2, column=0, columnspan=5, sticky=W)

    def detection(self):
        MODEL_NAME = 'PedJointNet_weights'
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('data', 'pedestrian_headshoulder.pbtxt')
        NUM_CLASSES = 2
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        # ## Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        image_path = self.selectFileName
        img_open = Image.open(image_path)
        image_np = self.load_image_into_numpy_array(img_open)
        output_dict = self.run_inference_for_single_image(image_np, detection_graph)
        # Visualization of the results of a detection.
        image_detect = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            image_path,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=1,
        )
        image_detect1 = Image.fromarray(image_detect)
        img1 = image_detect1.resize((500, 500))
        img1 = ImageTk.PhotoImage(img1)
        self.L1.image = img1
        Label(image=img1).grid(row=2, column=1, columnspan=5, sticky=W)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

if __name__ == '__main__':
    app = Applications()
    app.root.mainloop()







