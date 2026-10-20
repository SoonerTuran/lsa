"""Latent-space operators (phase 2+).

Planned modules, each acting on decoder latents with the decoder frozen:
  pose.py    g(z, T)        -> latent of the shape transformed by T in Sim(3)
  union.py   f(z_1..z_n)    -> latent of min-union of the shapes
  deform.py  h(z, d)        -> latent of the shape deformed by parameters d
"""
