GLOBAL_IS_DEBUG = True

def debug(values, sep=" ", end="\n"):
    if GLOBAL_IS_DEBUG:
        print(values, sep=sep, end=end)