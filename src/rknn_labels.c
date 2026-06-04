/* Shared label loading utility for RKNN tensor decoders.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "rknn_labels.h"
#include <gst/gst.h>

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

/**
 * rknn_labels_load:
 * @path: path to a text file with one class label per line
 *
 * Reads labels line-by-line using GDataInputStream (matching the pattern
 * used by GStreamer 1.28's built-in tensor decoders). Each label is
 * interned as a GQuark for zero-allocation lookup at runtime.
 *
 * Returns: (transfer full): a GArray of GQuark, or NULL on error
 */
GArray *
rknn_labels_load (const gchar *path)
{
  GFile *file;
  GFileInputStream *file_stream;
  GDataInputStream *data_stream;
  GError *error = NULL;
  GArray *array;
  gchar *line;

  if (!path || path[0] == '\0')
    return NULL;

  file = g_file_new_for_path (path);
  file_stream = g_file_read (file, NULL, &error);
  g_object_unref (file);

  if (!file_stream) {
    GST_WARNING ("Cannot open labels file %s: %s", path, error->message);
    g_clear_error (&error);
    return NULL;
  }

  data_stream = g_data_input_stream_new (G_INPUT_STREAM (file_stream));
  g_object_unref (file_stream);

  array = g_array_new (FALSE, FALSE, sizeof (GQuark));

  while ((line =
          g_data_input_stream_read_line (data_stream, NULL, NULL, &error))) {
    g_strstrip (line);
    if (line[0] != '\0') {
      GQuark q = g_quark_from_string (line);
      g_array_append_val (array, q);
    }
    g_free (line);
  }

  g_object_unref (data_stream);

  if (error) {
    GST_WARNING ("Error reading labels file %s: %s", path, error->message);
    g_clear_error (&error);
  }

  GST_INFO ("Loaded %u labels from %s", array->len, path);
  return array;
}

/**
 * rknn_labels_get:
 * @labels: (nullable): GArray of GQuark from rknn_labels_load(), or NULL
 * @index: class index
 *
 * Returns: GQuark for the label, or 0 if index is out of range
 */
GQuark
rknn_labels_get (GArray *labels, guint index)
{
  if (!labels || index >= labels->len)
    return 0;

  return g_array_index (labels, GQuark, index);
}

/**
 * rknn_labels_get_default_coco:
 *
 * Returns a static GArray of GQuark for the 80 COCO class names.
 * The returned array is owned by this function — do not free.
 */
GArray *
rknn_labels_get_default_coco (void)
{
  static GArray *coco = NULL;

  if (g_once_init_enter (&coco)) {
    GArray *a = g_array_sized_new (FALSE, FALSE, sizeof (GQuark),
        G_N_ELEMENTS (coco_classes));

    for (guint i = 0; i < G_N_ELEMENTS (coco_classes); i++) {
      GQuark q = g_quark_from_static_string (coco_classes[i]);
      g_array_append_val (a, q);
    }

    g_once_init_leave (&coco, a);
  }

  return coco;
}
