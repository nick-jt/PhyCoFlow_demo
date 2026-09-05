"""Python port of the dataviz skill's palette validator (node is too old here).

Checks the categorical line colors used in the PoF uncertainty figure:
lightness band, chroma floor, CVD separation (Machado 2009 severity 1.0),
normal-vision floor, and WCAG contrast against the white print surface.
"""
import math

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],
               [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}


def hex2srgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h):
    return [s2lin(c) for c in hex2srgb(h)]


def rellum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([rellum(a), rellum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s]


def sim(h, k):
    r, g, b = lin(h)
    M = MACHADO[k]
    cl = lambda c: max(0.0, min(1.0, c))
    return [cl(M[i][0] * r + M[i][1] * g + M[i][2] * b) for i in range(3)]


def dE(h1, h2, kind=None):
    a = oklab_lin(sim(h1, kind) if kind else lin(h1))
    b = oklab_lin(sim(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def oklch(h):
    L, a, b = oklab_lin(lin(h))
    return L, math.hypot(a, b)


if __name__ == "__main__":
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # Ux, Uy, Uz, p
    surface = "#ffffff"
    print("Lightness (band 0.43-0.77):", [(c, round(oklch(c)[0], 3)) for c in pal])
    print("Chroma (floor 0.10):", [(c, round(oklch(c)[1], 3)) for c in pal])
    allp = [(i, j) for i in range(len(pal)) for j in range(i + 1, len(pal))]
    adj = [(i, i + 1) for i in range(len(pal) - 1)]
    for name, pl in [("adjacent", adj), ("all-pairs", allp)]:
        worst = min((min(dE(pal[i], pal[j], "protan"), dE(pal[i], pal[j], "deutan")),
                     pal[i], pal[j]) for i, j in pl)
        nworst = min((dE(pal[i], pal[j]), pal[i], pal[j]) for i, j in pl)
        print(f"{name}: worst CVD dE={worst[0]:.1f} ({worst[1]}~{worst[2]}), "
              f"worst normal dE={nworst[0]:.1f} ({nworst[1]}~{nworst[2]})")
    print("Contrast vs white:", [(c, round(contrast(c, surface), 2)) for c in pal])
