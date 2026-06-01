/* YOLOv5/v8/v11 Post-Processing
 *
 * Anchor decode, DFL decode, confidence filtering, and non-maximum
 * suppression for YOLO-family object detection models.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __YOLO_POSTPROCESS_H__
#define __YOLO_POSTPROCESS_H__

#include <glib.h>

G_BEGIN_DECLS

typedef struct {
  gfloat x1, y1, x2, y2;
  gfloat score;
  gint   class_id;
} YoloDetection;

/* YOLOv5 3-scale anchors (P3/8, P4/16, P5/32) */
#define YOLOV5_NUM_SCALES        3
#define YOLOV5_ANCHORS_PER_SCALE 3
#define YOLOV5_NUM_CLASSES       80

/* YOLOv8/v11 use anchor-free DFL decoding */
#define YOLOV8_NUM_SCALES        3
#define YOLOV8_MAX_DFL_LEN       32

typedef enum {
  YOLO_MODEL_UNKNOWN = 0,
  YOLO_MODEL_V5,
  YOLO_MODEL_V8,
} YoloModelType;

/**
 * yolo_detect_model_type:
 * @n_outputs: number of output tensors
 * @dims: array of [n_channels, grid_h, grid_w] per output (3 values each)
 *
 * Returns: detected YOLO model variant based on output tensor structure
 */
YoloModelType yolo_detect_model_type (guint n_outputs,
                                      const guint *dims);

/**
 * yolov5_postprocess:
 * @outputs: array of 3 float tensors (one per scale), NCHW layout,
 *           values already post-sigmoid from the RKNN runtime
 * @n_outputs: number of output tensors (must be 3)
 * @dims: array of [n_channels, grid_h, grid_w] per output (3 values each)
 * @model_w: model input width (e.g. 640)
 * @model_h: model input height (e.g. 640)
 * @conf_threshold: minimum objectness * class_prob to keep
 * @nms_threshold: IoU threshold for NMS
 * @out_detections: (out): allocated array of detections, caller must g_free()
 * @out_count: (out): number of detections after NMS
 *
 * Coordinates in out_detections are in model-input pixel space (0..model_w).
 */
void yolov5_postprocess (const gfloat **outputs,
                         guint n_outputs,
                         const guint *dims,
                         guint model_w,
                         guint model_h,
                         gfloat conf_threshold,
                         gfloat nms_threshold,
                         YoloDetection **out_detections,
                         guint *out_count);

/**
 * yolov8_postprocess:
 * @outputs: array of 6 or 9 float tensors, NCHW layout.
 *           6 outputs: [box0, score0, box1, score1, box2, score2]
 *           9 outputs: [box0, score0, sum0, box1, score1, sum1, ...]
 *           Box tensors: channels = dfl_len * 4.
 *           Score tensors: channels = n_classes, values post-sigmoid.
 *           Score-sum tensors (optional): channels = 1.
 * @n_outputs: number of output tensors (6 or 9)
 * @dims: array of [n_channels, grid_h, grid_w] per output (3 values each)
 * @model_w: model input width (e.g. 640)
 * @model_h: model input height (e.g. 640)
 * @conf_threshold: minimum class confidence to keep
 * @nms_threshold: IoU threshold for NMS
 * @out_detections: (out): allocated array of detections, caller must g_free()
 * @out_count: (out): number of detections after NMS
 *
 * Coordinates in out_detections are in model-input pixel space (0..model_w).
 */
void yolov8_postprocess (const gfloat **outputs,
                         guint n_outputs,
                         const guint *dims,
                         guint model_w,
                         guint model_h,
                         gfloat conf_threshold,
                         gfloat nms_threshold,
                         YoloDetection **out_detections,
                         guint *out_count);

G_END_DECLS

#endif /* __YOLO_POSTPROCESS_H__ */
