/* GStreamer RKNN Object Detector Element
 *
 * Reads GstRknnTensorMeta (attached by rknninference), runs
 * model-specific post-processing, and outputs GstAnalyticsODMtd
 * for downstream elements like objectdetectionoverlay.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifdef HAVE_GST_ANALYTICS

#include "gstrknnobjectdetector.h"
#include "gstrknntensormeta.h"
#include "yolo_postprocess.h"

#include <gst/analytics/analytics.h>
#include <gst/analytics/gstanalyticsobjectdetectionmtd.h>

GST_DEBUG_CATEGORY_STATIC (gst_rknn_od_debug);
#define GST_CAT_DEFAULT gst_rknn_od_debug

enum {
  PROP_0,
  PROP_CONF_THRESHOLD,
  PROP_NMS_THRESHOLD,
};

#define DEFAULT_CONF_THRESHOLD 0.25f
#define DEFAULT_NMS_THRESHOLD  0.45f

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw"));

G_DEFINE_TYPE (GstRknnObjectDetector, gst_rknn_object_detector,
    GST_TYPE_BASE_TRANSFORM);

static const gchar *coco_classes[] = {
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
  "truck", "boat", "traffic light", "fire hydrant", "stop sign",
  "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
  "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
  "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
  "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
  "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
  "couch", "potted plant", "bed", "dining table", "toilet", "tv",
  "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
  "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
  "scissors", "teddy bear", "hair drier", "toothbrush",
};

static void
gst_rknn_object_detector_set_property (GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec)
{
  GstRknnObjectDetector *self = GST_RKNN_OBJECT_DETECTOR (object);

  switch (prop_id) {
    case PROP_CONF_THRESHOLD:
      self->conf_threshold = g_value_get_float (value);
      break;
    case PROP_NMS_THRESHOLD:
      self->nms_threshold = g_value_get_float (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_rknn_object_detector_get_property (GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec)
{
  GstRknnObjectDetector *self = GST_RKNN_OBJECT_DETECTOR (object);

  switch (prop_id) {
    case PROP_CONF_THRESHOLD:
      g_value_set_float (value, self->conf_threshold);
      break;
    case PROP_NMS_THRESHOLD:
      g_value_set_float (value, self->nms_threshold);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean
gst_rknn_object_detector_set_caps (GstBaseTransform *trans,
    GstCaps *incaps, GstCaps *outcaps)
{
  GstRknnObjectDetector *self = GST_RKNN_OBJECT_DETECTOR (trans);
  GstVideoInfo info;

  if (!gst_video_info_from_caps (&info, incaps)) {
    GST_ERROR_OBJECT (self, "Failed to parse input caps");
    return FALSE;
  }

  self->video_width = GST_VIDEO_INFO_WIDTH (&info);
  self->video_height = GST_VIDEO_INFO_HEIGHT (&info);

  GST_INFO_OBJECT (self, "Video: %dx%d", self->video_width, self->video_height);

  return TRUE;
}

/* Extract [C, H, W] from tensor meta dims (handles 3D and 4D layouts) */
static gboolean
extract_chw (const GstRknnTensorInfo *info, guint *c, guint *h, guint *w)
{
  if (info->n_dims == 4) {
    /* NCHW: [1, C, H, W] */
    *c = info->dims[1];
    *h = info->dims[2];
    *w = info->dims[3];
    return TRUE;
  } else if (info->n_dims == 3) {
    *c = info->dims[0];
    *h = info->dims[1];
    *w = info->dims[2];
    return TRUE;
  }
  return FALSE;
}

/* Infer model input size from the first box tensor's grid.
 * For both v5 and v8+, the first output's grid corresponds to stride 8. */
static void
infer_model_size (const guint *dims, YoloModelType model_type,
    guint *model_w, guint *model_h)
{
  /* dims[1] = grid_h, dims[2] = grid_w of the first tensor (stride 8) */
  *model_w = dims[2] * 8;
  *model_h = dims[1] * 8;
}

static GstFlowReturn
gst_rknn_object_detector_transform_ip (GstBaseTransform *trans,
    GstBuffer *buf)
{
  GstRknnObjectDetector *self = GST_RKNN_OBJECT_DETECTOR (trans);
  GstRknnTensorMeta *tmeta;
  YoloDetection *detections = NULL;
  guint n_detections = 0;

  tmeta = gst_buffer_get_rknn_tensor_meta (buf);
  if (!tmeta) {
    GST_LOG_OBJECT (self, "No tensor meta on buffer, passing through");
    return GST_FLOW_OK;
  }

  /* Extract [C, H, W] per tensor */
  guint dims[GST_RKNN_MAX_OUTPUTS * 3];
  const gfloat *outputs[GST_RKNN_MAX_OUTPUTS];

  for (guint i = 0; i < tmeta->n_tensors; i++) {
    outputs[i] = (const gfloat *) tmeta->data[i];
    if (!extract_chw (&tmeta->info[i],
            &dims[i * 3], &dims[i * 3 + 1], &dims[i * 3 + 2])) {
      GST_WARNING_OBJECT (self, "Unexpected tensor dims count: %u",
          tmeta->info[i].n_dims);
      return GST_FLOW_OK;
    }

    GST_LOG_OBJECT (self, "tensor[%u]: %ux%ux%u (%u elems)",
        i, dims[i*3], dims[i*3+1], dims[i*3+2], tmeta->info[i].n_elems);
  }

  /* Auto-detect model type on first frame */
  if (self->model_type == YOLO_MODEL_UNKNOWN) {
    self->model_type = yolo_detect_model_type (tmeta->n_tensors, dims);
    if (self->model_type == YOLO_MODEL_UNKNOWN) {
      GST_WARNING_OBJECT (self,
          "Cannot detect YOLO model type from %u output tensors — skipping",
          tmeta->n_tensors);
      return GST_FLOW_OK;
    }
    GST_INFO_OBJECT (self, "Detected model type: %s (%u outputs)",
        self->model_type == YOLO_MODEL_V5 ? "YOLOv5" : "YOLOv8/v11",
        tmeta->n_tensors);
  }

  guint model_w, model_h;
  infer_model_size (dims, self->model_type, &model_w, &model_h);
  GST_LOG_OBJECT (self, "Inferred model input: %ux%u", model_w, model_h);

  if (self->model_type == YOLO_MODEL_V5) {
    yolov5_postprocess (outputs, tmeta->n_tensors, dims,
        model_w, model_h,
        self->conf_threshold, self->nms_threshold,
        &detections, &n_detections);
  } else {
    yolov8_postprocess (outputs, tmeta->n_tensors, dims,
        model_w, model_h,
        self->conf_threshold, self->nms_threshold,
        &detections, &n_detections);
  }

  GST_LOG_OBJECT (self, "Decoded %u detections", n_detections);

  if (n_detections > 0 && self->video_width > 0) {
    GstAnalyticsRelationMeta *rmeta;
    rmeta = gst_buffer_add_analytics_relation_meta (buf);

    gfloat scale_x = (gfloat) self->video_width / (gfloat) model_w;
    gfloat scale_y = (gfloat) self->video_height / (gfloat) model_h;

    for (guint i = 0; i < n_detections; i++) {
      gint x = (gint) (detections[i].x1 * scale_x);
      gint y = (gint) (detections[i].y1 * scale_y);
      gint w = (gint) ((detections[i].x2 - detections[i].x1) * scale_x);
      gint h = (gint) ((detections[i].y2 - detections[i].y1) * scale_y);

      GQuark obj_type = 0;
      if (detections[i].class_id >= 0 &&
          detections[i].class_id < (gint) G_N_ELEMENTS (coco_classes)) {
        obj_type = g_quark_from_string (
            coco_classes[detections[i].class_id]);
      }

      GstAnalyticsODMtd od_mtd;
      gst_analytics_relation_meta_add_od_mtd (rmeta,
          obj_type, x, y, w, h, detections[i].score, &od_mtd);

      GST_DEBUG_OBJECT (self, "  [%u] %s %.2f @ (%d,%d %dx%d)",
          i,
          detections[i].class_id < (gint) G_N_ELEMENTS (coco_classes)
              ? coco_classes[detections[i].class_id] : "?",
          detections[i].score, x, y, w, h);
    }
  }

  g_free (detections);

  return GST_FLOW_OK;
}

static void
gst_rknn_object_detector_class_init (GstRknnObjectDetectorClass *klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *bt_class = GST_BASE_TRANSFORM_CLASS (klass);

  GST_DEBUG_CATEGORY_INIT (gst_rknn_od_debug, "rknnobjectdetector", 0,
      "RKNN object detection decoder");

  gobject_class->set_property = gst_rknn_object_detector_set_property;
  gobject_class->get_property = gst_rknn_object_detector_get_property;

  g_object_class_install_property (gobject_class, PROP_CONF_THRESHOLD,
      g_param_spec_float ("conf-threshold", "Confidence Threshold",
          "Minimum detection confidence (objectness x class probability)",
          0.0f, 1.0f, DEFAULT_CONF_THRESHOLD,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_NMS_THRESHOLD,
      g_param_spec_float ("nms-threshold", "NMS Threshold",
          "IoU threshold for non-maximum suppression",
          0.0f, 1.0f, DEFAULT_NMS_THRESHOLD,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata (element_class,
      "RKNN Object Detector",
      "Filter/Analyzer/Video",
      "Decodes RKNN inference tensors into object detection metadata",
      "Kelvin Lawson <info@lisden.com>");

  gst_element_class_add_static_pad_template (element_class, &sink_template);
  gst_element_class_add_static_pad_template (element_class, &src_template);

  bt_class->set_caps = gst_rknn_object_detector_set_caps;
  bt_class->transform_ip = gst_rknn_object_detector_transform_ip;

  /* Passthrough: we only add metadata, never modify the video */
  bt_class->passthrough_on_same_caps = TRUE;
}

static void
gst_rknn_object_detector_init (GstRknnObjectDetector *self)
{
  self->conf_threshold = DEFAULT_CONF_THRESHOLD;
  self->nms_threshold = DEFAULT_NMS_THRESHOLD;
  self->video_width = 0;
  self->video_height = 0;
  self->model_type = YOLO_MODEL_UNKNOWN;

  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (self), TRUE);
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (self), TRUE);
}

#endif /* HAVE_GST_ANALYTICS */
