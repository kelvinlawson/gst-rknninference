/* GStreamer RKNN Object Detector Element
 *
 * Decodes raw inference tensors into object detection metadata.
 * Supports YOLOv5 (anchor-based) and YOLOv8/v11 (DFL) output formats.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __GST_RKNN_OBJECT_DETECTOR_H__
#define __GST_RKNN_OBJECT_DETECTOR_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include "yolo_postprocess.h"

G_BEGIN_DECLS

#define GST_TYPE_RKNN_OBJECT_DETECTOR \
  (gst_rknn_object_detector_get_type ())

G_DECLARE_FINAL_TYPE (GstRknnObjectDetector, gst_rknn_object_detector,
    GST, RKNN_OBJECT_DETECTOR, GstBaseTransform)

struct _GstRknnObjectDetector {
  GstBaseTransform parent;

  /* Properties */
  gfloat  conf_threshold;
  gfloat  nms_threshold;

  /* From caps negotiation */
  gint    video_width;
  gint    video_height;

  /* Detected on first frame from tensor shape */
  YoloModelType model_type;
};

G_END_DECLS

#endif /* __GST_RKNN_OBJECT_DETECTOR_H__ */
