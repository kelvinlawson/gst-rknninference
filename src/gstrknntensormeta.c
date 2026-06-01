/* GStreamer RKNN Tensor Metadata
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "gstrknntensormeta.h"

static gboolean
gst_rknn_tensor_meta_init (GstMeta *meta, gpointer params,
    GstBuffer *buffer)
{
  GstRknnTensorMeta *tmeta = (GstRknnTensorMeta *) meta;
  tmeta->n_tensors = 0;
  memset (tmeta->info, 0, sizeof (tmeta->info));
  memset (tmeta->data, 0, sizeof (tmeta->data));
  return TRUE;
}

static void
gst_rknn_tensor_meta_free (GstMeta *meta, GstBuffer *buffer)
{
  GstRknnTensorMeta *tmeta = (GstRknnTensorMeta *) meta;

  for (guint i = 0; i < tmeta->n_tensors; i++) {
    g_free (tmeta->data[i]);
    tmeta->data[i] = NULL;
  }
  tmeta->n_tensors = 0;
}

/* Transform is not supported — tensor metadata is not meaningful
 * after scaling/cropping the video frame it was generated from. */
static gboolean
gst_rknn_tensor_meta_transform (GstBuffer *dest, GstMeta *meta,
    GstBuffer *buffer, GQuark type, gpointer data)
{
  return FALSE;
}

GType
gst_rknn_tensor_meta_api_get_type (void)
{
  static GType type = 0;
  static const gchar *tags[] = { NULL };

  if (g_once_init_enter (&type)) {
    GType _type =
        gst_meta_api_type_register ("GstRknnTensorMetaAPI", tags);
    g_once_init_leave (&type, _type);
  }
  return type;
}

const GstMetaInfo *
gst_rknn_tensor_meta_get_info (void)
{
  static const GstMetaInfo *meta_info = NULL;

  if (g_once_init_enter ((GstMetaInfo **) &meta_info)) {
    const GstMetaInfo *mi = gst_meta_register (
        GST_RKNN_TENSOR_META_API_TYPE,
        "GstRknnTensorMeta",
        sizeof (GstRknnTensorMeta),
        gst_rknn_tensor_meta_init,
        gst_rknn_tensor_meta_free,
        gst_rknn_tensor_meta_transform);
    g_once_init_leave ((GstMetaInfo **) &meta_info, (GstMetaInfo *) mi);
  }
  return meta_info;
}

GstRknnTensorMeta *
gst_buffer_add_rknn_tensor_meta (GstBuffer *buffer,
    guint n_tensors, const GstRknnTensorInfo *info, gpointer *data)
{
  GstRknnTensorMeta *meta;

  g_return_val_if_fail (GST_IS_BUFFER (buffer), NULL);
  g_return_val_if_fail (n_tensors <= GST_RKNN_MAX_OUTPUTS, NULL);

  meta = (GstRknnTensorMeta *) gst_buffer_add_meta (buffer,
      GST_RKNN_TENSOR_META_INFO, NULL);

  meta->n_tensors = n_tensors;
  for (guint i = 0; i < n_tensors; i++) {
    meta->info[i] = info[i];
    /* Take ownership of the data pointer */
    meta->data[i] = data[i];
  }

  return meta;
}

GstRknnTensorMeta *
gst_buffer_get_rknn_tensor_meta (GstBuffer *buffer)
{
  return (GstRknnTensorMeta *) gst_buffer_get_meta (buffer,
      GST_RKNN_TENSOR_META_API_TYPE);
}
