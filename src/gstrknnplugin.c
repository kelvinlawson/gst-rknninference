/* GStreamer RKNN Inference Plugin Registration
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include <gst/gst.h>
#include "gstrknninference.h"
#ifdef HAVE_GST_ANALYTICS
#include "gstrknnobjectdetector.h"
#include "gstrknnclassifier.h"
#endif

static gboolean
plugin_init (GstPlugin *plugin)
{
  gboolean ret = TRUE;

  ret &= gst_element_register (plugin, "rknninference",
      GST_RANK_NONE, GST_TYPE_RKNN_INFERENCE);

#ifdef HAVE_GST_ANALYTICS
  ret &= gst_element_register (plugin, "rknnobjectdetector",
      GST_RANK_NONE, GST_TYPE_RKNN_OBJECT_DETECTOR);
  ret &= gst_element_register (plugin, "rknnclassifier",
      GST_RANK_NONE, GST_TYPE_RKNN_CLASSIFIER);
#endif

  return ret;
}

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    rknninference,
    "RKNN neural network inference for Rockchip RK3588 NPU",
    plugin_init,
    "0.1.0",
    "LGPL",
    "gst-rknninference",
    "https://github.com/kelvinlawson/gst-rknninference"
)
