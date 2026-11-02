"""Avatar support: event bus (daemon side, standard library only), speech timeline, renderer.

The daemon may import ``bus`` and ``timeline`` only; the renderer modules need the
``homeai[avatar]`` extra (pygame-ce, moderngl) and must never be imported by the daemon.
"""
