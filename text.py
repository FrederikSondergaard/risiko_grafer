import matplotlib.font_manager as fm

fonts = sorted(set(f.name for f in fm.fontManager.ttflist if "Times" in f.name))
print(fonts)
