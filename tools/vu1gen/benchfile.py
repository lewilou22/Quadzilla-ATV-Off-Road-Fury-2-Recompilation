"""Reader for the VU1 workload file written by ps2_vu1_bench.inc (ATV_VU_BENCH_CAPTURE)."""
import struct

STATE_OFFSET = 8          # VU1Interpreter: m_unit, then VU1State (8-aligned: it holds uint64 cycles)
PC_IN_STATE = 512 + 64 + 16 + 12 + 4


def read_samples(path):
    """Yields (code_bytes, start_pc) per captured run; code is shared between runs until it changes."""
    with open(path, 'rb') as f:
        blob = f.read()
    magic = blob[:8]
    if magic[:7] != b'ATVVUB1':
        raise ValueError('not a VU bench file')
    object_size, skip_begin, skip_end, packet_begin, code_size, data_size, count = struct.unpack_from('<7I', blob, 8)
    pos = 36
    code = None
    packet_end = packet_begin + 0x10000
    for _ in range(count):
        pos += 4  # maxCycles
        new_code = blob[pos]
        pos += 1
        if new_code:
            code = blob[pos:pos + code_size]
            pos += code_size
        pos += data_size
        head = blob[pos:pos + skip_begin]
        pos += skip_begin
        pos += packet_begin - skip_end
        packet_bytes = struct.unpack_from('<I', blob, pos)[0]
        pos += 4 + packet_bytes
        pos += object_size - packet_end
        pc = struct.unpack_from('<I', head, STATE_OFFSET + PC_IN_STATE)[0]
        yield code, pc
