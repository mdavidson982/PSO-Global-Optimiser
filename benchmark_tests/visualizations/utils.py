def _make_printer(do_print: bool):
    def _printer(str: str, end="\n"):
        print(str, end=end)
    def _noneprinter(str: str, end = "\n"):
        pass

    if do_print:
        return _printer
    else:
        return _noneprinter