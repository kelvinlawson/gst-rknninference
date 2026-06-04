/* RKNN YOLO v8-11 Tensor Decoder
 *
 * Decodes raw RKNN inference tensors into object detection metadata
 * for YOLOv8, v10, and v11 models (anchor-free, DFL box decoding).
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __GST_RKNN_YOLOV8_TENSOR_DEC_H__
#define __GST_RKNN_YOLOV8_TENSOR_DEC_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

G_BEGIN_DECLS

#define GST_TYPE_RKNN_YOLOV8_TENSOR_DEC \
  (gst_rknn_yolov8_tensor_dec_get_type ())

G_DECLARE_FINAL_TYPE (GstRknnYolov8TensorDec, gst_rknn_yolov8_tensor_dec,
    GST, RKNN_YOLOV8_TENSOR_DEC, GstBaseTransform)

struct _GstRknnYolov8TensorDec {
  GstBaseTransform parent;

  gfloat  cls_confi_thresh;
  gfloat  iou_thresh;
  gsize   max_detection;
  gchar  *label_file;
  GArray *labels;

  GstVideoInfo video_info;
};

G_END_DECLS

#endif /* __GST_RKNN_YOLOV8_TENSOR_DEC_H__ */
