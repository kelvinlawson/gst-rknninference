plugin_sources = [
  'gstrknnplugin.c',
  'gstrknninference.c',
  'gstrknntensormeta.c',
  'rknn_wrapper.c',
]

m_dep = meson.get_compiler('c').find_library('m', required : true)

plugin_deps = [
  gst_dep,
  gst_base_dep,
  gst_video_dep,
  gst_allocators_dep,
  rknn_dep,
  m_dep,
]

plugin_c_args = [
  '-Wno-unused-parameter',
  '-DPACKAGE="gst-rknninference"',
  '-DVERSION="' + meson.project_version() + '"',
]

if gst_analytics_dep.found()
  plugin_deps += gst_analytics_dep
  plugin_c_args += '-DHAVE_GST_ANALYTICS'
  plugin_sources += ['gstrknnobjectdetector.c', 'yolo_postprocess.c', 'gstrknnclassifier.c']
endif

if rga_dep.found()
  plugin_deps += rga_dep
  plugin_c_args += '-DHAVE_RGA'
  plugin_sources += 'rga_preprocess.c'
endif

gstrknninference = shared_library('gstrknninference',
  plugin_sources,
  c_args : plugin_c_args,
  include_directories : rknn_inc,
  dependencies : plugin_deps,
  install : true,
  install_dir : plugins_install_dir,
)
