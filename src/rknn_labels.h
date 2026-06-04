/* Shared label loading utility for RKNN tensor decoders.
 *
 * Modelled on GStreamer 1.28's read_labels() pattern: loads a text file
 * (one class name per line) into a GArray of GQuark for O(1) per-frame
 * lookup with zero string allocation.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __RKNN_LABELS_H__
#define __RKNN_LABELS_H__

#include <glib.h>
#include <gio/gio.h>

G_BEGIN_DECLS

GArray *rknn_labels_load (const gchar *path);

GQuark rknn_labels_get (GArray *labels, guint index);

GArray *rknn_labels_get_default_coco (void);

G_END_DECLS

#endif /* __RKNN_LABELS_H__ */
