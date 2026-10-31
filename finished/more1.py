"""One table per codebase: does the gain hold beyond transformers?"""
import domains as d
for lib in ("pandas", "scipy", "sklearn", "sympy", "networkx"):
    P = d.files(d.SITE + lib + "/**/*.py"); A, B, C = d.parts(P); d.run("code " + lib, A, B, C)
