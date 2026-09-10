"""Socket syscall denial in addition to the network namespace."""


def install_socket_filter():
    import ctypes, platform

    if platform.machine() != "x86_64":
        raise RuntimeError("Socket sandbox currently supports Linux x86_64 only")

    class Filter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint),
        ]

    class Program(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    # Validate AUDIT_ARCH_X86_64 before interpreting syscall numbers; reject compat and x32 ABIs.
    rules = (Filter * 10)(
        Filter(0x20, 0, 0, 4),
        Filter(0x15, 1, 0, 0xC000003E),
        Filter(0x06, 0, 0, 0x00050001),
        Filter(0x20, 0, 0, 0),
        Filter(0x35, 0, 1, 0x40000000),
        Filter(0x06, 0, 0, 0x00050001),
        Filter(0x15, 1, 0, 41),
        Filter(0x15, 0, 1, 53),
        Filter(0x06, 0, 0, 0x00050001),
        Filter(0x06, 0, 0, 0x7FFF0000),
    )
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) or libc.prctl(
        22, 2, ctypes.byref(Program(10, rules))
    ):
        raise OSError(ctypes.get_errno(), "seccomp failed")
