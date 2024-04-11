
PSO_LOG = 3
MPSO_LOG = 2
ITERATION_LOG = 1

def _make_printer(do_print: int):
    def _printer(str: str, end="\n"):
        print(str, end=end)
    def _noneprinter(str: str, end = "\n"):
        pass

    if do_print > 0:
        return _printer
    else:
        return _noneprinter