project('gst-rknninference', 'c',
  version : '0.1.0',
  license : 'LGPL-2.1-or-later',
  meson_version : '>= 0.62',
  default_options : [
    'warning_level=2',
    'c_std=c11',
  ],
)

gst_dep = dependency('gstreamer-1.0', version : '>= 1.24')
gst_base_dep = dependency('gstreamer-base-1.0', version : '>= 1.24')
gst_video_dep = dependency('gstreamer-video-1.0', version : '>= 1.24')
gst_analytics_dep = dependency('gstreamer-analytics-1.0', version : '>= 1.24',
  required : false)
gst_allocators_dep = dependency('gstreamer-allocators-1.0', version : '>= 1.24')

cc = meson.get_compiler('c')

rknn_dep = cc.find_library('rknnrt', required : true)
rknn_inc = include_directories('/usr/local/include')

rga_dep = dependency('librga', required : false)
if not rga_dep.found()
  rga_dep = cc.find_library('rga', required : false)
endif

plugins_install_dir = join_paths(
  get_option('libdir'), 'gstreamer-1.0')

subdir('src')

endif
