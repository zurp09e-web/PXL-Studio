"""PXL v8 experimental lossless engine with legacy v4-v7 readers.

New v6-v8 streams use PXL-owned tokenization and rANS. Pillow is used only for
image I/O; NumPy/optional Numba/CuPy are compute dependencies.
"""
from __future__ import annotations

import binascii
import copy
import math
import os
import struct
from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

import pxl as core

VERSION = 8
MAGIC = core.MAGIC
HEADER = core.HEADER
TILE_HEADER = core.TILE_HEADER
PROFILE_LOSSLESS = core.PROFILE_LOSSLESS
PROFILE_LOSSY = core.PROFILE_LOSSY
CODEC_DCT = core.CODEC_DCT
CODEC_RAW_PIXELS = core.CODEC_RAW_PIXELS
CODEC_PRED2D = 20
CODEC_PIXEL_LZ = 21
CODEC_BLOCK2D = 22
CODEC_PALETTE = 23
CODEC_PRED2D_SIGNED = 24
CODEC_PRED2D_TRIPLE = 25
CODEC_PRED2D_MIXED = 26
CODEC_BLOCK2D_VAR = 27
V8_TILE_HEADER = struct.Struct("<IIHHBBIII")  # x,y,w,h,codec,transform,payload,decoded,crc
V8_GROUP_INTERVAL = 8
# The encoder accepts checkpoint intervals up to 256; reject archives that
# claim longer history groups before buffering unbounded tile payloads.
V8_GROUP_MAX_TILES = 256

# Limits for the cross-tile reference state. These are part of v4 semantics.
PIXEL_HISTORY_BYTES = 512 * 1024
PIXEL_HASH_CHAIN = 10
PIXEL_HASH_LIMIT = 240_000
BLOCK_SIZE = 8
BLOCK_HISTORY_COUNT = 32_768
MAX_TOKEN_STREAM = core.MAX_TILE_PAYLOAD

Progress = Optional[Callable[[int, str], None]]

# The v5 stream adds optional chunked entropy framing. These kernels compile PXL's own
# prediction/context algorithms with Numba; no external compressor is used.
if core._numba_njit is not None:
    @core._numba_njit(cache=True)
    def _pixel_hash_numba(data, offset, channels):
        h = np.uint64(1469598103934665603)
        prime = np.uint64(1099511628211)
        for j in range(3 * channels):
            h = (h ^ np.uint64(data[offset + j])) * prime
        if h == 0:
            h = np.uint64(1)
        return h

    @core._numba_njit(cache=True)
    def _pixel_hash_slot_numba(key, keys, mask):
        mask_i = np.int64(mask)
        slot = np.int64(key % np.uint64(mask_i + 1))
        while keys[slot] != 0 and keys[slot] != key:
            slot += 1
            if slot > mask_i:
                slot = 0
        return slot

    @core._numba_njit(cache=True)
    def _pixel_hash_insert_numba(data, base_abs, absolute, channels, window,
                                 keys, heads, links, hashes, positions, mask):
        offset = (absolute - base_abs) * channels
        key = _pixel_hash_numba(data, offset, channels)
        slot = _pixel_hash_slot_numba(key, keys, mask)
        ring = absolute % window
        links[ring] = heads[slot]
        hashes[ring] = key
        positions[ring] = absolute
        if keys[slot] == 0:
            keys[slot] = key
        heads[slot] = absolute

    @core._numba_njit(cache=True)
    def _pixel_hash_rebuild_numba(keys, heads, links, hashes, positions,
                                  low_abs, high_abs, window, mask):
        keys[:] = 0
        heads[:] = -1
        for absolute in range(low_abs, high_abs):
            ring = absolute % window
            if positions[ring] != absolute or hashes[ring] == 0:
                continue
            key = hashes[ring]
            slot = _pixel_hash_slot_numba(key, keys, mask)
            if keys[slot] == 0:
                keys[slot] = key
            links[ring] = heads[slot]
            heads[slot] = absolute

    @core._numba_njit(cache=True)
    def _pixel_lz_read_varint(tokens, pos):
        value = 0
        shift = 0
        for i in range(10):
            if pos >= tokens.size:
                return 0, pos, -1
            b = int(tokens[pos])
            pos += 1
            # The tenth byte of a uint64 varint may contain only bit zero.
            if i == 9 and b > 1:
                return 0, pos, -2
            value |= (b & 127) << shift
            if (b & 128) == 0:
                return value, pos, 0
            shift += 7
        return 0, pos, -2

    @core._numba_njit(cache=True)
    def _pixel_lz_decode_numba(tokens, history, expected_pixels, channels):
        output = np.empty(expected_pixels * channels, dtype=np.uint8)
        out_pos = 0
        pos = 0
        while pos < tokens.size:
            op = int(tokens[pos])
            pos += 1
            if op == 0:
                count, pos, err = _pixel_lz_read_varint(tokens, pos)
                if err != 0 or count < 1:
                    return np.empty(0, dtype=np.uint8), -1
                n = count * channels
                if pos + n > tokens.size or out_pos + n > output.size:
                    return np.empty(0, dtype=np.uint8), -1
                output[out_pos:out_pos + n] = tokens[pos:pos + n]
                pos += n
                out_pos += n
            elif op == 1:
                count, pos, err = _pixel_lz_read_varint(tokens, pos)
                if err != 0 or count < 3 or pos + channels > tokens.size:
                    return np.empty(0, dtype=np.uint8), -1
                if out_pos + count * channels > output.size:
                    return np.empty(0, dtype=np.uint8), -1
                for i in range(count):
                    for c in range(channels):
                        output[out_pos + i * channels + c] = tokens[pos + c]
                pos += channels
                out_pos += count * channels
            elif op == 2:
                distance, pos, err = _pixel_lz_read_varint(tokens, pos)
                if err != 0:
                    return np.empty(0, dtype=np.uint8), -1
                count, pos, err = _pixel_lz_read_varint(tokens, pos)
                if err != 0 or count < 4:
                    return np.empty(0, dtype=np.uint8), -1
                produced = out_pos // channels
                available = history.size // channels + produced
                if distance < 1 or distance > available or out_pos + count * channels > output.size:
                    return np.empty(0, dtype=np.uint8), -1
                for i in range(count):
                    for c in range(channels):
                        source = history.size + out_pos - distance * channels
                        if source < history.size:
                            output[out_pos] = history[source]
                        else:
                            local_source = source - history.size
                            if local_source < 0 or local_source >= out_pos:
                                return np.empty(0, dtype=np.uint8), -1
                            output[out_pos] = output[local_source]
                        out_pos += 1
            else:
                return np.empty(0, dtype=np.uint8), -1
        if out_pos != output.size:
            return np.empty(0, dtype=np.uint8), -1
        return output, 0

    @core._numba_njit(cache=True)
    def _pixel_lz_write_varint(out, pos, value):
        while value >= 128:
            out[pos] = (value & 127) | 128
            pos += 1
            value >>= 7
        out[pos] = value
        return pos + 1

    @core._numba_njit(cache=True)
    def _pixel_lz_varint_size(value):
        n = 1
        while value >= 128:
            n += 1
            value >>= 7
        return n

    @core._numba_njit(cache=True)
    def _pixel_lz_index_range_numba(data, base_abs, total_pixels, start_abs,
                                    end_abs, channels, window, keys, heads,
                                    links, hashes, positions, mask):
        low = max(base_abs, total_pixels - window)
        start = max(start_abs, low)
        end = min(end_abs, base_abs + data.size // channels - 2)
        inserted = 0
        for absolute in range(start, end):
            _pixel_hash_insert_numba(data, base_abs, absolute, channels, window,
                                     keys, heads, links, hashes, positions, mask)
            inserted += 1
        return max(start_abs, end), inserted

    @core._numba_njit(cache=True)
    def _pixel_lz_encode_numba(raw, history, total_pixels, indexed_until,
                               channels, window, keys, heads, links, hashes,
                               positions, mask, since_rebuild):
        combined = np.empty(history.size + raw.size, dtype=np.uint8)
        combined[:history.size] = history
        combined[history.size:] = raw
        prefix_pixels = history.size // channels
        base_abs = total_pixels - prefix_pixels
        pixels = raw.size // channels
        end_abs = total_pixels + pixels
        # Keep the open-addressed hash table bounded on very large images.
        if since_rebuild > window // 2:
            low_abs = max(base_abs, total_pixels - window)
            _pixel_hash_rebuild_numba(keys, heads, links, hashes, positions,
                                      low_abs, indexed_until, window, mask)
            since_rebuild = 0

        # Output is bounded by the literal bytes plus a conservative token allowance.
        out = np.empty(raw.size + pixels * 8 + 64, dtype=np.uint8)
        out_pos = 0
        literal_start = 0
        literal_count = 0
        i = 0
        while i < pixels:
            target_abs = total_pixels + i
            indexed_until, inserted = _pixel_lz_index_range_numba(
                combined, base_abs, total_pixels, indexed_until, target_abs,
                channels, window, keys, heads, links, hashes, positions, mask)
            since_rebuild += inserted
            local = prefix_pixels * channels + i * channels
            best_len = 0
            best_dist = 0

            if i + 3 <= pixels:
                key = _pixel_hash_numba(combined, local, channels)
                slot = _pixel_hash_slot_numba(key, keys, mask)
                candidate = heads[slot]
                checked = 0
                max_dist = min(window, target_abs - base_abs)
                while candidate >= 0 and checked < 10:
                    ring = candidate % window
                    if positions[ring] != candidate:
                        break
                    distance = target_abs - candidate
                    if distance > 0 and distance <= max_dist and hashes[ring] == key:
                        src = (candidate - base_abs) * channels
                        length = 0
                        limit = pixels - i
                        match = True
                        while length < limit and match:
                            for c in range(channels):
                                if combined[src + length * channels + c] != combined[local + length * channels + c]:
                                    match = False
                                    break
                            if match:
                                length += 1
                        if length > best_len:
                            best_len = length
                            best_dist = distance
                            if length >= 128:
                                break
                    candidate = links[ring]
                    checked += 1

            run = 1
            while i + run < pixels and run < 0x7fffffff:
                same = True
                a = local + (run - 1) * channels
                b = local + run * channels
                for c in range(channels):
                    if combined[a + c] != combined[b + c]:
                        same = False
                        break
                if not same:
                    break
                run += 1

            match_cost = 1 + _pixel_lz_varint_size(best_dist) + _pixel_lz_varint_size(best_len)
            run_cost = 1 + channels + _pixel_lz_varint_size(run)
            literal_cost = best_len * channels
            use_match = best_len >= 4 and match_cost < literal_cost and best_len >= run
            use_run = (not use_match) and run >= 3 and run_cost < run * channels

            if use_match or use_run:
                if literal_count:
                    out[out_pos] = 0
                    out_pos += 1
                    out_pos = _pixel_lz_write_varint(out, out_pos, literal_count)
                    src = prefix_pixels * channels + literal_start * channels
                    n = literal_count * channels
                    out[out_pos:out_pos + n] = combined[src:src + n]
                    out_pos += n
                    literal_count = 0
                if use_match:
                    out[out_pos] = 2
                    out_pos += 1
                    out_pos = _pixel_lz_write_varint(out, out_pos, best_dist)
                    out_pos = _pixel_lz_write_varint(out, out_pos, best_len)
                    i += best_len
                else:
                    out[out_pos] = 1
                    out_pos += 1
                    out_pos = _pixel_lz_write_varint(out, out_pos, run)
                    out[out_pos:out_pos + channels] = combined[local:local + channels]
                    out_pos += channels
                    i += run
            else:
                if literal_count == 0:
                    literal_start = i
                literal_count += 1
                i += 1

        if literal_count:
            out[out_pos] = 0
            out_pos += 1
            out_pos = _pixel_lz_write_varint(out, out_pos, literal_count)
            src = prefix_pixels * channels + literal_start * channels
            n = literal_count * channels
            out[out_pos:out_pos + n] = combined[src:src + n]
            out_pos += n

        indexed_until, inserted = _pixel_lz_index_range_numba(
            combined, base_abs, total_pixels, indexed_until, end_abs,
            channels, window, keys, heads, links, hashes, positions, mask)
        since_rebuild += inserted
        # Return a compact token buffer; state/history updates happen in Python.
        return out[:out_pos].copy(), indexed_until, since_rebuild

    @core._numba_njit(cache=True)
    def _context2d_ids_numba(data, width, height, channels):
        row_size = width * channels + channels
        ids = np.full(data.size, 16, dtype=np.uint8)
        for y in range(height):
            row0 = y * row_size
            residual0 = row0 + channels
            for x in range(width):
                pixel0 = residual0 + x * channels
                for ch in range(channels):
                    i = pixel0 + ch
                    left = int(data[i - channels]) if x else 0
                    up = int(data[i - row_size]) if y else 0
                    ids[i] = ((left >> 6) << 2) | (up >> 6)
        return ids

    @core._numba_njit(cache=True)
    def _filter_rows_v3_numba(source, width, height, channels):
        stride = width * channels
        row_size = stride + channels
        out = np.empty(height * row_size, dtype=np.uint8)
        for y in range(height):
            out_base = y * row_size
            src_base = y * stride
            # The predictor IDs are stored before the filtered row, one per channel.
            for ch in range(channels):
                out[out_base + ch] = 0
            for ch in range(channels):
                best_entropy = 1.0e300
                best_magnitude = 1 << 62
                best_pred = 0
                hist = np.zeros(256, dtype=np.int64)
                for pred in range(7):
                    hist[:] = 0
                    magnitude = 0
                    for x in range(width):
                        i = src_base + x * channels + ch
                        value = int(source[i])
                        left = int(source[i - channels + 0]) if x else 0
                        up = int(source[i - stride]) if y else 0
                        ul = int(source[i - stride - channels]) if y and x else 0
                        if pred == 0:
                            prediction = 0
                        elif pred == 1:
                            prediction = left
                        elif pred == 2:
                            prediction = up
                        elif pred == 3:
                            prediction = (left + up) // 2
                        elif pred == 4:
                            pp = left + up - ul
                            pa = abs(pp - left)
                            pb = abs(pp - up)
                            pc = abs(pp - ul)
                            if pa <= pb and pa <= pc:
                                prediction = left
                            elif pb <= pc:
                                prediction = up
                            else:
                                prediction = ul
                        elif pred == 5:
                            prediction = left + up - ul
                        else:
                            lo = min(left, up)
                            hi = max(left, up)
                            if ul >= hi:
                                prediction = lo
                            elif ul <= lo:
                                prediction = hi
                            else:
                                prediction = left + up - ul
                        residual = (value - prediction) & 255
                        hist[residual] += 1
                        magnitude += residual if residual <= 128 else 256 - residual
                    entropy_term = 0.0
                    for symbol in range(256):
                        count = hist[symbol]
                        if count > 0:
                            entropy_term += count * np.log2(count)
                    entropy = width * np.log2(width) - entropy_term
                    if entropy < best_entropy or (entropy == best_entropy and magnitude < best_magnitude):
                        best_entropy = entropy
                        best_magnitude = magnitude
                        best_pred = pred
                out[out_base + ch] = best_pred
                # Reconstruct the winning residuals into the interleaved row.
                for x in range(width):
                    i = src_base + x * channels + ch
                    value = int(source[i])
                    left = int(source[i - channels]) if x else 0
                    up = int(source[i - stride]) if y else 0
                    ul = int(source[i - stride - channels]) if y and x else 0
                    pred = best_pred
                    if pred == 0:
                        prediction = 0
                    elif pred == 1:
                        prediction = left
                    elif pred == 2:
                        prediction = up
                    elif pred == 3:
                        prediction = (left + up) // 2
                    elif pred == 4:
                        pp = left + up - ul
                        pa = abs(pp - left)
                        pb = abs(pp - up)
                        pc = abs(pp - ul)
                        if pa <= pb and pa <= pc:
                            prediction = left
                        elif pb <= pc:
                            prediction = up
                        else:
                            prediction = ul
                    elif pred == 5:
                        prediction = left + up - ul
                    else:
                        lo = min(left, up)
                        hi = max(left, up)
                        if ul >= hi:
                            prediction = lo
                        elif ul <= lo:
                            prediction = hi
                        else:
                            prediction = left + up - ul
                    out[out_base + channels + x * channels + ch] = (value - prediction) & 255
        return out

    @core._numba_njit(cache=True)
    def _unfilter_rows_v3_numba(payload, width, height, channels):
        stride = width * channels
        row_size = stride + channels
        out = np.empty(height * stride, dtype=np.uint8)
        for y in range(height):
            src_base = y * row_size
            dst_base = y * stride
            for x in range(width):
                for ch in range(channels):
                    pred = int(payload[src_base + ch])
                    i = dst_base + x * channels + ch
                    left = int(out[i - channels]) if x else 0
                    up = int(out[i - stride]) if y else 0
                    ul = int(out[i - stride - channels]) if y and x else 0
                    if pred == 0:
                        prediction = 0
                    elif pred == 1:
                        prediction = left
                    elif pred == 2:
                        prediction = up
                    elif pred == 3:
                        prediction = (left + up) // 2
                    elif pred == 4:
                        pp = left + up - ul
                        pa = abs(pp - left)
                        pb = abs(pp - up)
                        pc = abs(pp - ul)
                        if pa <= pb and pa <= pc:
                            prediction = left
                        elif pb <= pc:
                            prediction = up
                        else:
                            prediction = ul
                    elif pred == 5:
                        prediction = left + up - ul
                    elif pred == 6:
                        lo = min(left, up)
                        hi = max(left, up)
                        if ul >= hi:
                            prediction = lo
                        elif ul <= lo:
                            prediction = hi
                        else:
                            prediction = left + up - ul
                    else:
                        return np.empty(0, dtype=np.uint8)
                    residual = int(payload[src_base + channels + x * channels + ch])
                    out[i] = (residual + prediction) & 255
        return out

    @core._numba_njit(cache=True)
    def _rans_context2d_decode_numba(blob, start_pos, output_size, width, height,
                                     channels, frequencies, cumulative, decode):
        state = np.uint64(0)
        for j in range(4):
            state |= np.uint64(blob[start_pos + j]) << np.uint64(8 * j)
        if state < np.uint64(core._RANS_L):
            return np.empty(0, dtype=np.uint8), -2
        pos = start_pos + 4
        out = np.empty(output_size, dtype=np.uint8)
        row_size = width * channels + channels
        for i in range(output_size):
            row_offset = i % row_size
            if row_offset < channels:
                ctx = 16
            else:
                sample_offset = row_offset - channels
                x = sample_offset // channels
                left = int(out[i - channels]) if x else 0
                up = int(out[i - row_size]) if i >= row_size else 0
                ctx = ((left >> 6) << 2) | (up >> 6)
            slot = int(state & np.uint64(core._RANS_TOTAL - 1))
            sym = int(decode[ctx, slot])
            freq = int(frequencies[ctx, sym])
            if freq <= 0:
                return np.empty(0, dtype=np.uint8), -2
            state = np.uint64(freq * (int(state) >> core._RANS_SCALE_BITS)
                              + slot - int(cumulative[ctx, sym]))
            while state < np.uint64(core._RANS_L):
                if pos >= blob.size:
                    return np.empty(0, dtype=np.uint8), -1
                state = (state << np.uint64(8)) | np.uint64(blob[pos])
                pos += 1
            out[i] = sym
        return out, pos


def _filter_rows_v3_fast(data: bytes, width: int, height: int, channels: int) -> bytes:
    if core._numba_njit is not None and len(data) >= 1024:
        source = np.frombuffer(data, dtype=np.uint8)
        return _filter_rows_v3_numba(source, width, height, channels).tobytes()
    return core._filter_rows_v3(data, width, height, channels)


def _unfilter_rows_v3_fast(payload: bytes, width: int, height: int, channels: int) -> bytes:
    if core._numba_njit is not None and len(payload) >= 1024:
        source = np.frombuffer(payload, dtype=np.uint8)
        expected = height * width * channels
        out = _unfilter_rows_v3_numba(source, width, height, channels)
        if out.size != expected:
            raise ValueError("Invalid predictor in PXL v4 stream")
        return out.tobytes()
    return core._unfilter_rows_v3(payload, width, height, channels)


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint cannot be negative")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for i in range(10):
        if pos >= len(data):
            raise ValueError("Truncated PXL v4 varint")
        b = data[pos]
        pos += 1
        # A uint64 varint has only one usable bit in its tenth byte.
        if i == 9 and b > 1:
            raise ValueError("Overflowing PXL v4 varint")
        value |= (b & 0x7F) << (i * 7)
        if not (b & 0x80):
            return value, pos
    raise ValueError("Oversized PXL v4 varint")


def _context2d_ids(data: bytes, width: int, height: int, channels: int) -> np.ndarray:
    """17 contexts: 16 left/up residual buckets and one row-header context."""
    row_size = width * channels + channels
    if len(data) != row_size * height:
        raise ValueError("PXL v4 predictor stream geometry mismatch")
    a = np.frombuffer(data, dtype=np.uint8)
    if core._numba_njit is not None and a.size >= 1024:
        return _context2d_ids_numba(a, width, height, channels)
    ids = np.full(a.size, 16, dtype=np.uint8)
    for y in range(height):
        row0 = y * row_size
        residual0 = row0 + channels
        for x in range(width):
            pixel0 = residual0 + x * channels
            for ch in range(channels):
                i = pixel0 + ch
                left = int(a[i - channels]) if x else 0
                up = int(a[i - row_size]) if y else 0
                ids[i] = ((left >> 6) << 2) | (up >> 6)
    return ids


def _rans_context2d_encode(data: bytes, width: int, height: int, channels: int,
                           context_ids=None, context_count: int = 17) -> bytes:
    """PXL-owned rANS over a supplied spatial context map."""
    if not data:
        raise ValueError("Cannot encode an empty PXL spatial rANS stream")
    ids = (_context2d_ids(data, width, height, channels) if context_ids is None
           else np.ascontiguousarray(context_ids, dtype=np.uint8).reshape(-1))
    count = int(context_count)
    if count not in (17, 65):
        raise ValueError("Unsupported PXL spatial context count")
    if ids.size != len(data) or (ids.size and int(ids.max()) >= count):
        raise ValueError("Invalid PXL spatial context ID array")
    arr = np.frombuffer(data, dtype=np.uint8)
    hist = np.bincount(ids.astype(np.int64) * 256 + arr.astype(np.int64),
                       minlength=count * 256).reshape(count, 256)
    freqs = np.zeros((count, 256), dtype=np.uint64)
    cumul = np.zeros((count, 256), dtype=np.uint64)
    table = bytearray((count,))
    for ctx in range(count):
        if not hist[ctx].any():
            table.extend(b"\x00\x00")
            continue
        f = core._normalise_frequency_counts(hist[ctx])
        freqs[ctx] = np.asarray(f, dtype=np.uint64)
        c = 0
        for sym, value in enumerate(f):
            cumul[ctx, sym] = c
            c += value
        packed = core._serialize_rans_table(f)
        table.extend(len(packed).to_bytes(2, "little"))
        table.extend(packed)
    native = None
    try:
        from pxl_native import get_backend as _get_native_backend
        native = _get_native_backend(False)
    except Exception:
        native = None
    if native is not None:
        state, emitted_bytes = native.rans_encode_ids(data, ids, count, freqs, cumul)
    elif core._numba_njit is not None:
        state, emitted = core._rans_context_ids_encode_kernel(arr, ids, freqs, cumul)
        emitted_bytes = emitted.tobytes()
    else:
        state = core._RANS_L
        reverse = bytearray()
        for i in range(arr.size - 1, -1, -1):
            ctx, sym = int(ids[i]), int(arr[i])
            f = int(freqs[ctx, sym])
            x_max = ((core._RANS_L >> core._RANS_SCALE_BITS) << 8) * f
            while state >= x_max:
                reverse.append(state & 255)
                state >>= 8
            state = ((state // f) << core._RANS_SCALE_BITS) + (state % f) + int(cumul[ctx, sym])
        emitted_bytes = bytes(reversed(reverse))
    return bytes(table) + int(state).to_bytes(4, "little") + emitted_bytes


def _context2d_signed_ids(data: bytes, width: int, height: int, channels: int) -> np.ndarray:
    """17 contexts using exact small signed residual classes.

    For each residual, 0, +1 and -1 get dedicated classes; all other values
    share class 3. The context is left_class * 4 + up_class. Row predictor
    selector bytes always use context 16. This is fully derivable by decoders.
    """
    row_size = width * channels + channels
    if len(data) != row_size * height:
        raise ValueError("PXL v6/v7 signed-context geometry mismatch")
    rows = np.frombuffer(data, dtype=np.uint8).reshape(height, row_size)
    residual = rows[:, channels:].reshape(height, width, channels)
    left = np.zeros_like(residual)
    up = np.zeros_like(residual)
    if width > 1:
        left[:, 1:, :] = residual[:, :-1, :]
    if height > 1:
        up[1:, :, :] = residual[:-1, :, :]
    classes = np.full(256, 3, dtype=np.uint8)
    classes[0] = 0
    classes[1] = 1
    classes[255] = 2
    ids = np.full((height, row_size), 16, dtype=np.uint8)
    ids[:, channels:] = (classes[left] * 4 + classes[up]).reshape(height, width * channels)
    return ids.reshape(-1)


def _context2d_triple_ids(data: bytes, width: int, height: int, channels: int) -> np.ndarray:
    """65 contexts: classes of left, up and upper-left residual neighbours."""
    row_size = width * channels + channels
    if len(data) != row_size * height:
        raise ValueError("PXL v6/v7 triple-context geometry mismatch")
    rows = np.frombuffer(data, dtype=np.uint8).reshape(height, row_size)
    residual = rows[:, channels:].reshape(height, width, channels)
    left = np.zeros_like(residual)
    up = np.zeros_like(residual)
    upper_left = np.zeros_like(residual)
    if width > 1:
        left[:, 1:, :] = residual[:, :-1, :]
    if height > 1:
        up[1:, :, :] = residual[:-1, :, :]
    if width > 1 and height > 1:
        upper_left[1:, 1:, :] = residual[:-1, :-1, :]
    classes = np.full(256, 3, dtype=np.uint8)
    classes[0] = 0
    classes[1] = 1
    classes[255] = 2
    ids = np.full((height, row_size), 64, dtype=np.uint8)
    ids[:, channels:] = (classes[left] * 16 + classes[up] * 4 +
                         classes[upper_left]).reshape(height, width * channels)
    return ids.reshape(-1)


def _context2d_mixed_ids(data: bytes, width: int, height: int, channels: int) -> np.ndarray:
    """Channel-specific 17-way contexts used by the blended probability model."""
    row_size = width * channels + channels
    base = _context2d_signed_ids(data, width, height, channels).reshape(height, row_size)
    columns = np.arange(row_size, dtype=np.int64)
    channel_ids = np.where(columns < channels, columns,
                           (columns - channels) % channels)
    return (channel_ids[None, :] * 17 + base).astype(np.uint8).reshape(-1)


def _rans_context2d_mixed_encode(data: bytes, width: int, height: int, channels: int) -> bytes:
    """rANS with local/channel/global histogram mixing, native-first."""
    try:
        from pxl_native import get_backend as _get_native_backend
        native = _get_native_backend(False)
    except Exception:
        native = None
    if native is not None:
        return native.mixed_rans_encode(data, width, height, channels)
    """Portable reference implementation used when the C++ library is absent.

    For each decodable context, the frequency counts are mixed as
    4*local + 2*channel + global. This backs off smoothly when a local context
    has few observations. The resulting static tables remain fully derivable
    from the residual stream and require no hidden adaptive state.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    count = channels * 17
    ids = _context2d_mixed_ids(data, width, height, channels).astype(np.int64)
    hist_ctx = np.bincount(ids * 256 + arr.astype(np.int64),
                           minlength=count * 256).reshape(count, 256)
    row_size = width * channels + channels
    cols = np.arange(row_size, dtype=np.int64)
    channel_ids = np.where(cols < channels, cols, (cols - channels) % channels)
    channel_ids = np.broadcast_to(channel_ids, (height, row_size)).reshape(-1)
    hist_channel = np.bincount(channel_ids * 256 + arr.astype(np.int64),
                               minlength=channels * 256).reshape(channels, 256)
    hist_global = np.bincount(arr, minlength=256)
    freqs = np.zeros((count, 256), dtype=np.uint64)
    cumul = np.zeros((count, 256), dtype=np.uint64)
    tables = bytearray((count,))
    for ctx in range(count):
        ch = ctx // 17
        mixed = hist_ctx[ctx] * 4 + hist_channel[ch] * 2 + hist_global
        if not mixed.any():
            tables.extend(b"\x00\x00")
            continue
        f = core._normalise_frequency_counts(mixed)
        freqs[ctx] = np.asarray(f, dtype=np.uint64)
        c = 0
        for sym, value in enumerate(f):
            cumul[ctx, sym] = c
            c += value
        packed = core._serialize_rans_table(f)
        tables.extend(len(packed).to_bytes(2, "little"))
        tables.extend(packed)
    state = core._RANS_L
    emitted = bytearray()
    for i in range(arr.size - 1, -1, -1):
        ctx, sym = int(ids[i]), int(arr[i])
        f = int(freqs[ctx, sym])
        x_max = ((core._RANS_L >> core._RANS_SCALE_BITS) << 8) * f
        while state >= x_max:
            emitted.append(state & 255)
            state >>= 8
        state = ((state // f) << core._RANS_SCALE_BITS) + (state % f) + int(cumul[ctx, sym])
    return bytes(tables) + int(state).to_bytes(4, "little") + bytes(reversed(emitted))


def _rans_context2d_mixed_decode(blob: bytes, output_size: int, width: int, height: int,
                                channels: int) -> bytes:
    try:
        from pxl_native import get_backend as _get_native_backend
        native = _get_native_backend(False)
    except Exception:
        native = None
    if native is not None:
        return native.mixed_rans_decode(blob, output_size, width, height, channels)
    count = channels * 17
    row_size = width * channels + channels
    if len(blob) < 1 + count * 2 + 4 or blob[0] != count:
        raise ValueError("Invalid PXL mixed-context rANS header")
    if output_size != height * row_size or output_size > core.MAX_TILE_PAYLOAD:
        raise ValueError("Invalid PXL mixed-context output geometry")
    pos = 1
    freqs = np.zeros((count, 256), dtype=np.uint64)
    cumul = np.zeros((count, 256), dtype=np.uint64)
    decode = np.zeros((count, core._RANS_TOTAL), dtype=np.uint8)
    for ctx in range(count):
        if pos + 2 > len(blob):
            raise ValueError("Truncated PXL mixed-context table length")
        size = int.from_bytes(blob[pos:pos + 2], "little")
        pos += 2
        if size == 0:
            continue
        if pos + size > len(blob):
            raise ValueError("Truncated PXL mixed-context table")
        f, end = core._parse_rans_table(blob[:pos + size], pos)
        if end != pos + size:
            raise ValueError("Invalid PXL mixed-context table size")
        pos += size
        freqs[ctx] = np.asarray(f, dtype=np.uint64)
        c = 0
        for sym, value in enumerate(f):
            cumul[ctx, sym] = c
            if value:
                decode[ctx, c:c + value] = sym
            c += value
    if pos + 4 > len(blob):
        raise ValueError("Truncated PXL mixed-context rANS state")
    state = int.from_bytes(blob[pos:pos + 4], "little")
    pos += 4
    if state < core._RANS_L:
        raise ValueError("Invalid PXL mixed-context rANS state")
    out = bytearray(output_size)
    for i in range(output_size):
        col = i % row_size
        if col < channels:
            ch, base_ctx = col, 16
        else:
            sample_offset = col - channels
            x = sample_offset // channels
            ch = sample_offset % channels
            left = out[i - channels] if x else 0
            up = out[i - row_size] if i >= row_size else 0
            cl = 0 if left == 0 else (1 if left == 1 else (2 if left == 255 else 3))
            cu = 0 if up == 0 else (1 if up == 1 else (2 if up == 255 else 3))
            base_ctx = cl * 4 + cu
        ctx = ch * 17 + base_ctx
        slot = state & (core._RANS_TOTAL - 1)
        sym = int(decode[ctx, slot])
        f = int(freqs[ctx, sym])
        if f == 0:
            raise ValueError("Invalid PXL mixed-context model symbol")
        state = f * (state >> core._RANS_SCALE_BITS) + slot - int(cumul[ctx, sym])
        while state < core._RANS_L:
            if pos >= len(blob):
                raise ValueError("Truncated PXL mixed-context stream")
            state = (state << 8) | blob[pos]
            pos += 1
        out[i] = sym
    if pos != len(blob):
        raise ValueError("Trailing bytes in PXL mixed-context stream")
    return bytes(out)


def _rans_context2d_decode(blob: bytes, output_size: int, width: int, height: int,
                            channels: int, signed_context: bool = False,
                            triple_context: bool = False) -> bytes:
    if signed_context and triple_context:
        raise ValueError("Only one PXL spatial context model may be selected")
    count = 65 if triple_context else 17
    if len(blob) < 1 + count * 2 + 4 or blob[0] != count:
        raise ValueError("Invalid PXL v4 2D-context rANS header")
    if output_size != height * (width * channels + channels) or output_size > core.MAX_TILE_PAYLOAD:
        raise ValueError("Invalid PXL v4 2D-context output geometry")
    pos = 1
    freqs = np.zeros((count, 256), dtype=np.uint64)
    cumul = np.zeros((count, 256), dtype=np.uint64)
    decode = np.zeros((count, core._RANS_TOTAL), dtype=np.uint8)
    for ctx in range(count):
        if pos + 2 > len(blob):
            raise ValueError("Truncated PXL v4 context table length")
        size = int.from_bytes(blob[pos:pos + 2], "little")
        pos += 2
        if size == 0:
            continue
        if pos + size > len(blob):
            raise ValueError("Truncated PXL v4 context table")
        f, end = core._parse_rans_table(blob[:pos + size], pos)
        if end != pos + size:
            raise ValueError("Invalid PXL v4 context table size")
        pos += size
        freqs[ctx] = np.asarray(f, dtype=np.uint64)
        c = 0
        for sym, value in enumerate(f):
            cumul[ctx, sym] = c
            if value:
                decode[ctx, c:c + value] = sym
            c += value
    if pos + 4 > len(blob):
        raise ValueError("Truncated PXL v4 rANS state")
    state = int.from_bytes(blob[pos:pos + 4], "little")
    if state < core._RANS_L:
        raise ValueError("Invalid PXL v4 rANS state")
    native = None
    try:
        from pxl_native import get_backend as _get_native_backend
        native = _get_native_backend(False)
    except Exception:
        native = None
    if native is not None and output_size >= 64:
        if triple_context:
            decoded, end_pos = native.rans_decode_context2d_triple(
                blob, pos, output_size, width, height, channels, freqs, cumul, decode)
        elif signed_context:
            decoded, end_pos = native.rans_decode_context2d_signed(
                blob, pos, output_size, width, height, channels, freqs, cumul, decode)
        else:
            decoded, end_pos = native.rans_decode_context2d(
                blob, pos, output_size, width, height, channels, freqs, cumul, decode)
        if end_pos != len(blob):
            raise ValueError("Trailing bytes in PXL 2D-context rANS payload")
        return decoded
    if not signed_context and not triple_context and core._numba_njit is not None and output_size >= 1024:
        freq_array = freqs.astype(np.uint32)
        cumul_array = cumul.astype(np.uint32)
        decode_array = decode
        decoded, end_pos = _rans_context2d_decode_numba(
            np.frombuffer(blob, dtype=np.uint8), pos, output_size, width, height,
            channels, freq_array, cumul_array, decode_array)
        if end_pos == -1:
            raise ValueError("Truncated PXL v4 rANS payload")
        if end_pos == -2:
            raise ValueError("Invalid PXL v4 rANS model symbol")
        if end_pos != len(blob):
            raise ValueError("Trailing bytes in PXL v4 rANS payload")
        return decoded.tobytes()
    pos += 4
    out = bytearray(output_size)
    row_size = width * channels + channels
    for i in range(output_size):
        row_offset = i % row_size
        if row_offset < channels:
            ctx = 64 if triple_context else 16
        else:
            sample_offset = row_offset - channels
            x = sample_offset // channels
            left = out[i - channels] if x else 0
            up = out[i - row_size] if i >= row_size else 0
            if triple_context:
                upper_left = out[i - row_size - channels] if (x and i >= row_size) else 0
                left_class = 0 if left == 0 else (1 if left == 1 else (2 if left == 255 else 3))
                up_class = 0 if up == 0 else (1 if up == 1 else (2 if up == 255 else 3))
                ul_class = 0 if upper_left == 0 else (1 if upper_left == 1 else (2 if upper_left == 255 else 3))
                ctx = left_class * 16 + up_class * 4 + ul_class
            elif signed_context:
                left_class = 0 if left == 0 else (1 if left == 1 else (2 if left == 255 else 3))
                up_class = 0 if up == 0 else (1 if up == 1 else (2 if up == 255 else 3))
                ctx = left_class * 4 + up_class
            else:
                ctx = ((left >> 6) << 2) | (up >> 6)
        slot = state & (core._RANS_TOTAL - 1)
        sym = int(decode[ctx, slot])
        f = int(freqs[ctx, sym])
        if f == 0:
            raise ValueError("Invalid PXL v4 rANS model symbol")
        state = f * (state >> core._RANS_SCALE_BITS) + slot - int(cumul[ctx, sym])
        while state < core._RANS_L:
            if pos >= len(blob):
                raise ValueError("Truncated PXL v4 rANS payload")
            state = (state << 8) | blob[pos]
            pos += 1
        out[i] = sym
    if pos != len(blob):
        raise ValueError("Trailing bytes in PXL v4 rANS payload")
    return bytes(out)


class PixelLZState:
    """Pixel-aware LZ with a bounded cross-tile history.

    The encoder's Numba hash table is allocated lazily: decode does not need an
    index at all, and raw/predictor-only images should not reserve dictionary RAM.
    """
    def __init__(self, channels: int, tile_size: int = 256):
        self.channels = channels
        self.tile_size = int(tile_size)
        self.window_pixels = max(3, PIXEL_HISTORY_BYTES // channels)
        self.history_limit = self.window_pixels * channels
        self.history = bytearray()
        self.total_pixels = 0
        self.indexed_until = 0
        self.heads: OrderedDict[bytes, deque[int]] = OrderedDict()
        try:
            from pxl_native import get_backend as _get_native_backend
            self._native_backend = _get_native_backend(False)
        except Exception:
            self._native_backend = None
        self._native_state = None  # lazily allocated persistent C++ encoder state
        self._fast = core._numba_njit is not None and self._native_backend is None
        self._hash_arrays_ready = False
        self._since_rebuild = 0
        self._hash_mask = 0
        self._hash_keys = None
        self._hash_heads = None
        self._ring_prev = None
        self._ring_hash = None
        self._ring_pos = None

    def __deepcopy__(self, memo):
        """Clone dictionary state without copying the process-wide ctypes handle.

        Lossy encoding evaluates pixel-LZ as a candidate on a trial state. The
        native backend is immutable/shared; only the history and Python index
        belong to an individual state instance.
        """
        clone = type(self)(self.channels, self.tile_size)
        memo[id(self)] = clone
        clone.history = bytearray(self.history)
        clone.total_pixels = self.total_pixels
        clone.indexed_until = self.indexed_until
        clone._since_rebuild = self._since_rebuild
        clone._native_state = None  # lazily rebuild the C++ index from cloned history
        if self._native_backend is None:
            clone.heads = copy.deepcopy(self.heads, memo)
            clone._hash_mask = self._hash_mask
            clone._hash_arrays_ready = self._hash_arrays_ready
            for name in ("_hash_keys", "_hash_heads", "_ring_prev", "_ring_hash", "_ring_pos"):
                value = getattr(self, name)
                setattr(clone, name, None if value is None else value.copy())
        return clone

    def _ensure_fast_arrays(self):
        if not self._fast or self._hash_arrays_ready:
            return
        table_size = 1
        capacity_hint = max(self.window_pixels * 4,
                            self.tile_size * self.tile_size * 2,
                            1 << 20)
        while table_size < capacity_hint:
            table_size <<= 1
        self._hash_mask = table_size - 1
        self._hash_keys = np.zeros(table_size, dtype=np.uint64)
        self._hash_heads = np.full(table_size, -1, dtype=np.int64)
        self._ring_prev = np.full(self.window_pixels, -1, dtype=np.int64)
        self._ring_hash = np.zeros(self.window_pixels, dtype=np.uint64)
        self._ring_pos = np.full(self.window_pixels, -1, dtype=np.int64)
        self._hash_arrays_ready = True

    def _insert_range(self, combined: bytes, base_pixel: int, start_abs: int, end_abs: int):
        min_abs = max(base_pixel, self.total_pixels - self.window_pixels)
        start_abs = max(start_abs, min_abs)
        end_abs = min(end_abs, base_pixel + len(combined) // self.channels - 2)
        ch = self.channels
        for absolute in range(start_abs, end_abs):
            local = (absolute - base_pixel) * ch
            key = combined[local:local + 3 * ch]
            if len(key) != 3 * ch:
                continue
            q = self.heads.get(key)
            if q is None:
                q = deque(maxlen=PIXEL_HASH_CHAIN)
                self.heads[key] = q
            elif isinstance(self.heads, OrderedDict):
                self.heads.move_to_end(key)
            q.append(absolute)
        self.indexed_until = max(self.indexed_until, end_abs)
        while len(self.heads) > PIXEL_HASH_LIMIT:
            self.heads.popitem(last=False)

    def update(self, raw: bytes, index: bool = True):
        ch = self.channels
        if len(raw) % ch:
            raise ValueError("PXL v5 pixel history is not pixel aligned")
        pixels = len(raw) // ch
        combined = bytes(self.history) + raw
        base = self.total_pixels - len(self.history) // ch
        end_total = self.total_pixels + pixels
        if self._native_backend is not None:
            if self._native_state is not None:
                self._native_state.update(raw)
            self.total_pixels = end_total
            self.history = bytearray(combined[-self.history_limit:])
            return
        if index:
            if self._fast:
                self._ensure_fast_arrays()
                data = np.frombuffer(combined, dtype=np.uint8)
                self.indexed_until, inserted = _pixel_lz_index_range_numba(
                    data, base, self.total_pixels, self.indexed_until, end_total,
                    ch, self.window_pixels, self._hash_keys, self._hash_heads,
                    self._ring_prev, self._ring_hash, self._ring_pos, self._hash_mask)
                self._since_rebuild += int(inserted)
                if self._since_rebuild > self.window_pixels // 2:
                    low = max(base, end_total - self.window_pixels)
                    _pixel_hash_rebuild_numba(
                        self._hash_keys, self._hash_heads, self._ring_prev,
                        self._ring_hash, self._ring_pos, low, self.indexed_until,
                        self.window_pixels, self._hash_mask)
                    self._since_rebuild = 0
            else:
                self._insert_range(combined, base, self.indexed_until, end_total)
        self.total_pixels = end_total
        self.history = bytearray(combined[-self.history_limit:])

    def encode(self, raw: bytes) -> bytes:
        ch = self.channels
        if self._native_backend is not None:
            if len(raw) % ch:
                raise ValueError("PXL v5 tile data is not pixel aligned")
            if self._native_state is None:
                self._native_state = self._native_backend.new_pixel_lz_state(ch, self.window_pixels)
                self._native_state.seed(bytes(self.history), self.total_pixels)
            tokens = self._native_state.encode(raw)
            combined = bytes(self.history) + raw
            self.total_pixels += len(raw) // ch
            self.history = bytearray(combined[-self.history_limit:])
            return tokens
        if self._fast:
            if len(raw) % ch:
                raise ValueError("PXL v5 tile data is not pixel aligned")
            self._ensure_fast_arrays()
            combined_history = np.frombuffer(bytes(self.history), dtype=np.uint8)
            source = np.frombuffer(raw, dtype=np.uint8)
            token_array, self.indexed_until, self._since_rebuild = _pixel_lz_encode_numba(
                source, combined_history, self.total_pixels, self.indexed_until,
                ch, self.window_pixels, self._hash_keys, self._hash_heads,
                self._ring_prev, self._ring_hash, self._ring_pos, self._hash_mask,
                self._since_rebuild)
            combined = bytes(self.history) + raw
            self.total_pixels += len(raw) // ch
            self.history = bytearray(combined[-self.history_limit:])
            if self._since_rebuild > self.window_pixels // 2:
                low = max(self.total_pixels - len(self.history) // ch,
                          self.total_pixels - self.window_pixels)
                _pixel_hash_rebuild_numba(
                    self._hash_keys, self._hash_heads, self._ring_prev,
                    self._ring_hash, self._ring_pos, low, self.indexed_until,
                    self.window_pixels, self._hash_mask)
                self._since_rebuild = 0
            return token_array.tobytes()
        if len(raw) % ch:
            raise ValueError("PXL v5 tile data is not pixel aligned")
        pixels = len(raw) // ch
        history = bytes(self.history)
        combined = history + raw
        base_abs = self.total_pixels - len(history) // ch
        tile_start = self.total_pixels
        end_abs = tile_start + pixels
        prefix_pixels = len(history) // ch
        out = bytearray()
        literals = bytearray()
        literal_pixels = 0

        def flush_literals():
            nonlocal literal_pixels
            if literal_pixels:
                out.append(0)
                out.extend(_varint(literal_pixels))
                out.extend(literals)
                literals.clear()
                literal_pixels = 0

        i = 0
        while i < pixels:
            target_abs = tile_start + i
            self._insert_range(combined, base_abs, self.indexed_until, target_abs)
            local = prefix_pixels * ch + i * ch
            best_len = 0
            best_dist = 0
            if i + 3 <= pixels:
                key = combined[local:local + 3 * ch]
                chain = self.heads.get(key, ())
                max_dist = min(self.window_pixels, target_abs - base_abs)
                for candidate in reversed(chain):
                    distance = target_abs - candidate
                    if distance <= 0 or distance > max_dist:
                        continue
                    src = (candidate - base_abs) * ch
                    length = 0
                    limit = pixels - i
                    while length < limit:
                        s = src + length * ch
                        t = local + length * ch
                        if combined[s:s + ch] != combined[t:t + ch]:
                            break
                        length += 1
                    if length > best_len:
                        best_len, best_dist = length, distance
                        if length >= 128:
                            break

            run = 1
            while i + run < pixels and run < 0x7fffffff:
                a = local + (run - 1) * ch
                b = local + run * ch
                if combined[a:a + ch] != combined[b:b + ch]:
                    break
                run += 1

            match_cost = 1 + len(_varint(best_dist)) + len(_varint(best_len))
            run_cost = 1 + ch + len(_varint(run))
            literal_cost = best_len * ch
            if best_len >= 4 and match_cost < literal_cost and best_len >= run:
                flush_literals()
                out.append(2)
                out.extend(_varint(best_dist))
                out.extend(_varint(best_len))
                i += best_len
            elif run >= 3 and run_cost < run * ch:
                flush_literals()
                out.append(1)
                out.extend(_varint(run))
                out.extend(combined[local:local + ch])
                i += run
            else:
                literals.extend(combined[local:local + ch])
                literal_pixels += 1
                i += 1
        flush_literals()

        # Index the just-consumed tile and update the bounded byte history.
        self._insert_range(combined, base_abs, self.indexed_until, end_abs)
        self.total_pixels = end_abs
        self.history = bytearray(combined[-self.history_limit:])
        return bytes(out)

    def decode(self, tokens: bytes, expected_pixels: int) -> bytes:
        ch = self.channels
        if self._native_backend is not None:
            raw = self._native_backend.pixel_lz_decode(tokens, bytes(self.history), expected_pixels, ch)
            self.update(raw, index=False)
            return raw
        if self._fast:
            decoded, status = _pixel_lz_decode_numba(
                np.frombuffer(tokens, dtype=np.uint8),
                np.frombuffer(bytes(self.history), dtype=np.uint8),
                int(expected_pixels), ch)
            if status != 0:
                raise ValueError("Invalid PXL v4 pixel token stream")
            raw = decoded.tobytes()
            self.update(raw, index=False)
            return raw
        out = bytearray()
        pos = 0
        while pos < len(tokens):
            op = tokens[pos]
            pos += 1
            if op == 0:
                count, pos = _read_varint(tokens, pos)
                n = count * ch
                if count < 1 or pos + n > len(tokens):
                    raise ValueError("Invalid PXL v4 literal token")
                out.extend(tokens[pos:pos + n])
                pos += n
            elif op == 1:
                count, pos = _read_varint(tokens, pos)
                if count < 3 or pos + ch > len(tokens):
                    raise ValueError("Invalid PXL v4 pixel-run token")
                pixel = tokens[pos:pos + ch]
                pos += ch
                out.extend(pixel * count)
            elif op == 2:
                distance, pos = _read_varint(tokens, pos)
                count, pos = _read_varint(tokens, pos)
                produced = len(out) // ch
                available = len(self.history) // ch + produced
                if distance < 1 or distance > available or count < 4:
                    raise ValueError("Invalid PXL v4 pixel-reference token")
                if count > expected_pixels - produced:
                    raise ValueError("PXL v4 pixel-reference exceeds tile output")
                for _ in range(count):
                    source = len(self.history) + len(out) - distance * ch
                    if source < len(self.history):
                        pixel = self.history[source:source + ch]
                    else:
                        local_source = source - len(self.history)
                        pixel = out[local_source:local_source + ch]
                    if len(pixel) != ch:
                        raise ValueError("Invalid PXL v4 reference source")
                    out.extend(pixel)
            else:
                raise ValueError("Unknown PXL v4 pixel token")
            if len(out) > expected_pixels * ch:
                raise ValueError("PXL v4 pixel token stream expands too far")
        if len(out) != expected_pixels * ch:
            raise ValueError("PXL v4 pixel token stream size mismatch")
        raw = bytes(out)
        # Decoding references only the retained pixel history; indexing is encoder-only.
        self.update(raw, index=False)
        return raw


class BlockReferenceState:
    """Exact 8x8 (edge-aware) block dictionary shared across all image tiles."""
    def __init__(self, channels: int):
        self.channels = channels
        self.next_id = 0
        self.blocks: OrderedDict[int, tuple[int, int, bytes]] = OrderedDict()
        self.heads: dict[tuple[int, int, bytes], deque[int]] = {}
        try:
            from pxl_native import get_backend as _get_native_backend
            native = _get_native_backend(False)
            self._native_state = native.new_block_state(channels, BLOCK_HISTORY_COUNT) if native else None
        except Exception:
            self._native_state = None

    def __deepcopy__(self, memo):
        clone = type(self)(self.channels)
        memo[id(self)] = clone
        if self._native_state is not None:
            clone._native_state.close()
            clone._native_state = self._native_state.clone()
        else:
            clone.next_id = self.next_id
            clone.blocks = copy.deepcopy(self.blocks, memo)
            clone.heads = copy.deepcopy(self.heads, memo)
        return clone

    def _key(self, bw: int, bh: int, block: bytes):
        # Hash only narrows candidates; full bytes are verified before a ref is used.
        import hashlib
        return bw, bh, hashlib.blake2b(block, digest_size=8).digest()

    def _add(self, bw: int, bh: int, block: bytes):
        idx = self.next_id
        self.next_id += 1
        self.blocks[idx] = (bw, bh, block)
        key = self._key(bw, bh, block)
        chain = self.heads.get(key)
        if chain is None:
            chain = deque(maxlen=12)
            self.heads[key] = chain
        chain.append(idx)
        while len(self.blocks) > BLOCK_HISTORY_COUNT:
            old_idx, _ = self.blocks.popitem(last=False)
            # Lazy pruning of candidate chains keeps the insertion hot path cheap.
            if old_idx % 4096 == 0:
                for k in list(self.heads.keys()):
                    q = self.heads[k]
                    while q and q[0] not in self.blocks:
                        q.popleft()
                    if not q:
                        del self.heads[k]

    def _iter_blocks(self, raw: bytes, width: int, height: int):
        ch = self.channels
        stride = width * ch
        for by in range(0, height, BLOCK_SIZE):
            bh = min(BLOCK_SIZE, height - by)
            for bx in range(0, width, BLOCK_SIZE):
                bw = min(BLOCK_SIZE, width - bx)
                block = bytearray(bw * bh * ch)
                p = 0
                for y in range(bh):
                    start = (by + y) * stride + bx * ch
                    n = bw * ch
                    block[p:p + n] = raw[start:start + n]
                    p += n
                yield bx, by, bw, bh, bytes(block)

    def encode(self, raw: bytes, width: int, height: int) -> bytes:
        if self._native_state is not None:
            return self._native_state.encode(raw, width, height)
        tokens = bytearray()
        for _, _, bw, bh, block in self._iter_blocks(raw, width, height):
            key = self._key(bw, bh, block)
            best = None
            for candidate in reversed(self.heads.get(key, ())):
                item = self.blocks.get(candidate)
                if item is not None and item == (bw, bh, block):
                    best = candidate
                    break
            if best is not None:
                tokens.append(1)
                tokens.extend(_varint(self.next_id - best))
            else:
                tokens.append(0)
                tokens.extend(block)
            self._add(bw, bh, block)
        return bytes(tokens)

    def update(self, raw: bytes, width: int, height: int):
        if self._native_state is not None:
            self._native_state.update(raw, width, height)
            return
        for _, _, bw, bh, block in self._iter_blocks(raw, width, height):
            self._add(bw, bh, block)

    def decode(self, tokens: bytes, width: int, height: int) -> bytes:
        if self._native_state is not None:
            return self._native_state.decode(tokens, width, height)
        out = bytearray(width * height * self.channels)
        pos = 0
        ch = self.channels
        count = 0
        for by in range(0, height, BLOCK_SIZE):
            bh = min(BLOCK_SIZE, height - by)
            for bx in range(0, width, BLOCK_SIZE):
                bw = min(BLOCK_SIZE, width - bx)
                if pos >= len(tokens):
                    raise ValueError("Truncated PXL v4 block token stream")
                op = tokens[pos]
                pos += 1
                if op == 0:
                    size = bw * bh * ch
                    if pos + size > len(tokens):
                        raise ValueError("Truncated PXL v4 literal block")
                    block = tokens[pos:pos + size]
                    pos += size
                elif op == 1:
                    distance, pos = _read_varint(tokens, pos)
                    source_id = self.next_id - distance
                    item = self.blocks.get(source_id)
                    if distance < 1 or item is None or item[0] != bw or item[1] != bh:
                        raise ValueError("Invalid PXL v4 block reference")
                    block = item[2]
                else:
                    raise ValueError("Unknown PXL v4 block operation")
                p = 0
                for y in range(bh):
                    dst = ((by + y) * width + bx) * ch
                    n = bw * ch
                    out[dst:dst + n] = block[p:p + n]
                    p += n
                self._add(bw, bh, block)
                count += 1
        if pos != len(tokens):
            raise ValueError("Trailing bytes in PXL v4 block token stream")
        return bytes(out)


def _encode_zero_delta(current: bytes, reference: bytes) -> bytes:
    """Encode bytewise modular differences as zero-run/literal-run pairs."""
    out = bytearray()
    pos = 0
    while pos < len(current):
        zero = 0
        while pos < len(current) and ((current[pos] - reference[pos]) & 255) == 0:
            zero += 1
            pos += 1
        start = pos
        while pos < len(current) and ((current[pos] - reference[pos]) & 255) != 0:
            pos += 1
        out.extend(_varint(zero))
        out.extend(_varint(pos - start))
        for i in range(start, pos):
            out.append((current[i] - reference[i]) & 255)
    return bytes(out)


def _decode_zero_delta(data: bytes, pos: int, reference: bytes):
    out = bytearray(len(reference))
    made = 0
    while made < len(reference):
        zero, pos = _read_varint(data, pos)
        literal, pos = _read_varint(data, pos)
        if zero == 0 and literal == 0:
            raise ValueError("Empty PXL delta-block run")
        if zero > len(reference) - made:
            raise ValueError("PXL delta-block zero run exceeds block")
        out[made:made + zero] = reference[made:made + zero]
        made += zero
        if literal > len(reference) - made or pos + literal > len(data):
            raise ValueError("PXL delta-block literal run exceeds stream")
        for i in range(literal):
            out[made] = (reference[made] + data[pos + i]) & 255
            made += 1
        pos += literal
    return bytes(out), pos


class VariableBlockReferenceState:
    """Adaptive exact block dictionary using 8x8, 16x16 and 32x32 matches.

    Large repeated regions are represented by one long-distance reference;
    unmatched regions remain small 8x8 literals. Hashes only narrow candidate
    lists: complete block bytes are always compared before a reference is used.
    """
    SIZES = (8, 16, 32)

    def __init__(self, channels: int, limit: int = 32768):
        self.channels = int(channels)
        self.limit = int(limit)
        self.next_id = 0
        self.blocks = OrderedDict()
        self.heads = {}
        self._native_state = None
        try:
            from pxl_native import get_backend as _get_native_backend
            backend = _get_native_backend(False)
            if backend is not None:
                self._native_state = backend.new_variable_block_state(self.channels, self.limit)
        except Exception:
            self._native_state = None

    def __deepcopy__(self, memo):
        clone = type(self)(self.channels, self.limit)
        memo[id(self)] = clone
        clone.next_id = self.next_id
        clone.blocks = copy.deepcopy(self.blocks, memo)
        clone.heads = copy.deepcopy(self.heads, memo)
        if self._native_state is not None:
            if clone._native_state is not None:
                clone._native_state.close()
            clone._native_state = self._native_state.clone()
        return clone

    @staticmethod
    def _hash(bw, bh, block):
        import hashlib
        return (bw, bh, hashlib.blake2b(block, digest_size=8).digest())

    def _add_one(self, bw, bh, block):
        ident = self.next_id
        self.next_id += 1
        key = self._hash(bw, bh, block)
        self.blocks[ident] = (bw, bh, block)
        self.heads.setdefault(key, deque()).append(ident)
        while len(self.blocks) > self.limit:
            old_id, old = self.blocks.popitem(last=False)
            old_key = self._hash(*old)
            q = self.heads.get(old_key)
            if q:
                while q and q[0] not in self.blocks:
                    q.popleft()
                if not q:
                    self.heads.pop(old_key, None)

    def _find(self, bw, bh, block):
        key = self._hash(bw, bh, block)
        for ident in reversed(self.heads.get(key, ())):
            if self.blocks.get(ident) == (bw, bh, block):
                return ident
        return None

    def _find_delta(self, bw, bh, block):
        """Find a recent same-sized block that yields a cheaper delta stream."""
        best = None
        current = np.frombuffer(block, dtype=np.uint8)
        searched = 0
        for ident, item in reversed(self.blocks.items()):
            if item[0] != bw or item[1] != bh:
                continue
            reference = item[2]
            if len(reference) != len(block):
                continue
            searched += 1
            if searched > 48:
                break
            ref_array = np.frombuffer(reference, dtype=np.uint8)
            equal = int(np.count_nonzero(current == ref_array))
            if equal * 5 < len(block) * 4:
                continue
            delta = _encode_zero_delta(block, reference)
            cost = 2 + len(_varint(self.next_id - ident)) + len(delta)
            if cost < len(block) and (best is None or cost < best[0]):
                best = (cost, ident, delta)
        return best

    def _extract(self, raw, width, x, y, bw, bh):
        ch = self.channels
        out = bytearray(bw * bh * ch)
        p = 0
        stride = width * ch
        for yy in range(bh):
            start = (y + yy) * stride + x * ch
            n = bw * ch
            out[p:p+n] = raw[start:start+n]
            p += n
        return bytes(out)

    def _add_region(self, bw, bh, block):
        # Store the whole match and all contained 16/8 sub-blocks so later tiles
        # can reference either a large region or a smaller repeated detail.
        self._add_one(bw, bh, block)
        ch = self.channels
        for size in (16, 8):
            if bw < size or bh < size:
                continue
            for y in range(0, bh, size):
                for x in range(0, bw, size):
                    sw, sh = min(size, bw-x), min(size, bh-y)
                    if sw != size or sh != size:
                        continue
                    sub = bytearray(sw * sh * ch)
                    p = 0
                    for yy in range(sh):
                        start = ((y + yy) * bw + x) * ch
                        n = sw * ch
                        sub[p:p+n] = block[start:start+n]
                        p += n
                    self._add_one(sw, sh, bytes(sub))

    def _grid(self, width, height):
        covered = set()
        for y in range(0, height, 8):
            for x in range(0, width, 8):
                cell = (x, y)
                if cell in covered:
                    continue
                yield x, y, covered

    def update(self, raw: bytes, width: int, height: int):
        if self._native_state is not None:
            self._native_state.update(raw, width, height)
            return
        # Canonical state transition when another codec was selected. Index
        # aligned 32/16/8 regions so future tiles can reference large matches
        # even when this tile itself did not use the variable-block codec.
        for size in (32, 16, 8):
            for y in range(0, height, size):
                for x in range(0, width, size):
                    bw, bh = min(size, width-x), min(size, height-y)
                    if size > 8 and (bw != size or bh != size):
                        continue
                    self._add_one(bw, bh, self._extract(raw, width, x, y, bw, bh))

    def has_large_matches(self, raw: bytes, width: int, height: int) -> bool:
        if self._native_state is not None:
            return self._native_state.has_large_matches(raw, width, height)
        """Cheaply screen whether 16x16/32x32 references can be used.

        The legacy 8x8 dictionary already handles small exact blocks. Avoid
        cloning/searching the larger dictionary unless a larger cross-tile
        match is actually present in this tile.
        """
        for size in (32, 16):
            if width < size or height < size:
                continue
            for y in range(0, height - size + 1, size):
                for x in range(0, width - size + 1, size):
                    block = self._extract(raw, width, x, y, size, size)
                    if self._find(size, size, block) is not None:
                        return True
                    if self._find_delta(size, size, block) is not None:
                        return True
        return False

    def encode(self, raw: bytes, width: int, height: int) -> bytes:
        if self._native_state is not None:
            return self._native_state.encode(raw, width, height)
        tokens = bytearray()
        for x, y, covered in self._grid(width, height):
            chosen = None
            best_saving = 0
            for size in (32, 16, 8):
                if x + size > width or y + size > height:
                    continue
                cells = [(xx, yy) for yy in range(y, y+size, 8)
                         for xx in range(x, x+size, 8)]
                if any(c in covered for c in cells):
                    continue
                block = self._extract(raw, width, x, y, size, size)
                exact_id = self._find(size, size, block)
                if exact_id is not None:
                    distance = self.next_id - exact_id
                    cost = 2 + len(_varint(distance))
                    saving = len(block) - cost
                    candidate = (saving, 1, size, block, distance, b"")
                    if saving > best_saving:
                        chosen, best_saving = candidate, saving
                    continue
                if size >= 16:
                    delta = self._find_delta(size, size, block)
                    if delta is not None:
                        cost, ident, delta_bytes = delta
                        distance = self.next_id - ident
                        saving = len(block) - cost
                        candidate = (saving, 2, size, block, distance, delta_bytes)
                        if saving > best_saving:
                            chosen, best_saving = candidate, saving
            if chosen is None:
                bw, bh = min(8, width-x), min(8, height-y)
                block = self._extract(raw, width, x, y, bw, bh)
                tokens.extend((0, 0))
                tokens.extend(block)
            else:
                _, opcode, size, block, distance, delta_bytes = chosen
                bw = bh = size
                size_code = self.SIZES.index(size)
                tokens.extend((opcode, size_code))
                tokens.extend(_varint(distance))
                if opcode == 2:
                    tokens.extend(delta_bytes)
            for yy in range(y, y+bh, 8):
                for xx in range(x, x+bw, 8):
                    covered.add((xx, yy))
            self._add_region(bw, bh, block)
        return bytes(tokens)

    def decode(self, tokens: bytes, width: int, height: int) -> bytes:
        if self._native_state is not None:
            return self._native_state.decode(tokens, width, height)
        out = bytearray(width * height * self.channels)
        pos = 0
        for x, y, covered in self._grid(width, height):
            if pos + 2 > len(tokens):
                raise ValueError("Truncated PXL variable-block token stream")
            op, size_code = tokens[pos], tokens[pos+1]
            pos += 2
            if size_code >= len(self.SIZES):
                raise ValueError("Invalid PXL variable-block size")
            size = self.SIZES[size_code]
            if op == 0:
                if size_code != 0:
                    raise ValueError("Variable-block literals must use the 8x8 base size")
                bw, bh = min(8, width-x), min(8, height-y)
                n = bw * bh * self.channels
                if pos + n > len(tokens):
                    raise ValueError("Truncated PXL variable-block literal")
                block = tokens[pos:pos+n]
                pos += n
            elif op in (1, 2):
                if x + size > width or y + size > height:
                    raise ValueError("Variable-block reference exceeds tile geometry")
                distance, pos = _read_varint(tokens, pos)
                ident = self.next_id - distance
                item = self.blocks.get(ident)
                if distance < 1 or item is None or item[0] != size or item[1] != size:
                    raise ValueError("Invalid PXL variable-block reference")
                bw = bh = size
                if op == 1:
                    block = item[2]
                else:
                    block, pos = _decode_zero_delta(tokens, pos, item[2])
            else:
                raise ValueError("Unknown PXL variable-block operation")
            ch = self.channels
            p = 0
            for yy in range(bh):
                dst = ((y + yy) * width + x) * ch
                n = bw * ch
                out[dst:dst+n] = block[p:p+n]
                p += n
            for yy in range(y, y+bh, 8):
                for xx in range(x, x+bw, 8):
                    covered.add((xx, yy))
            self._add_region(bw, bh, block)
        if pos != len(tokens):
            raise ValueError("Trailing bytes in PXL variable-block stream")
        return bytes(out)


def _is_flat_tile(raw: bytes, channels: int) -> bool:
    if not raw or len(raw) % channels:
        return False
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(-1, channels)
    return bool(np.all(pixels == pixels[0]))


def _tile_traits(raw: bytes, width: int, height: int, channels: int) -> tuple[bool, bool]:
    """Cheap entropy and spatial/repetition screening before expensive searches."""
    if not raw:
        return False, False
    entropy = core._entropy_cost_array(np.frombuffer(raw, dtype=np.uint8)) / len(raw)
    high_entropy = entropy > 7.95
    if high_entropy:
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, channels)
        residual_entropies = []
        if width > 1:
            hres = (pixels[:, 1:, :].astype(np.int16) - pixels[:, :-1, :].astype(np.int16)) & 255
            residual_entropies.append(core._entropy_cost_array(hres.astype(np.uint8).ravel()) / hres.size)
        if height > 1:
            vres = (pixels[1:, :, :].astype(np.int16) - pixels[:-1, :, :].astype(np.int16)) & 255
            residual_entropies.append(core._entropy_cost_array(vres.astype(np.uint8).ravel()) / vres.size)
        # A high raw histogram can still be a smooth gradient. Only skip the
        # predictor search when simple horizontal/vertical residuals stay noisy.
        if residual_entropies and min(residual_entropies) < 7.90:
            high_entropy = False
    if not high_entropy:
        return False, True
    pixel_count = len(raw) // channels
    sample_count = min(2048, max(0, pixel_count - 2))
    if sample_count < 8:
        return True, False
    step = max(1, pixel_count // sample_count)
    seen = set()
    limit = pixel_count - 2
    for pos in range(0, limit, step):
        start = pos * channels
        key = raw[start:start + 3 * channels]
        if key in seen:
            return True, True
        seen.add(key)
    return True, False


def _predictor_candidate(raw: bytes, width: int, height: int, channels: int,
                         effort: int, backend: str = "cpu",
                         parallel_transforms: bool = True) -> tuple[int, int, bytes] | None:
    """Search predictor transforms and both 2D context models by final size.

    v6 adds a signed-small-residual context model. Every candidate is fully
    entropy-coded before selection, so this search cannot make output larger
    than the best candidate it already considered in v5.
    """
    if effort <= 1:
        transforms = (core.TRANSFORM_NONE, core.TRANSFORM_G)
    elif effort >= 9:
        transforms = (core.TRANSFORM_NONE, core.TRANSFORM_G, core.TRANSFORM_R,
                      core.TRANSFORM_B, core.TRANSFORM_YCOCG)
    else:
        transforms = (core.TRANSFORM_NONE, core.TRANSFORM_G, core.TRANSFORM_YCOCG)
    native = None
    if backend != "cuda":
        try:
            from pxl_native import get_backend as _get_native_backend
            native = _get_native_backend(False)
        except Exception:
            native = None

    def evaluate(transform):
        context_ids = None
        if native is not None:
            filtered, context_ids = native.filter_color_v3(
                raw, width, height, channels, transform, with_contexts=True)
        elif backend == "cuda":
            transformed = core._forward_color_cuda(raw, channels, transform)
            # The CUDA row-filter implementation lives in pxl.py (imported as
            # ``core`` above). Calling it as a bare local name raised NameError
            # on real CUDA encodes even though the CuPy smoke test succeeded.
            filtered = core._filter_rows_cuda_v3(transformed, width, height, channels)
        else:
            transformed = core._forward_color(raw, channels, transform)
            filtered = _filter_rows_v3_fast(transformed, width, height, channels)
        encoded = _rans_context2d_encode(filtered, width, height, channels, context_ids)
        payload = struct.pack("<I", len(filtered)) + encoded
        return (len(payload), CODEC_PRED2D, transform, payload, filtered)

    # Color transforms are independent. ctypes releases the GIL for C++ kernels.
    if native is not None and parallel_transforms and len(raw) >= 65536 and len(transforms) > 1:
        with ThreadPoolExecutor(max_workers=len(transforms)) as pool:
            regular_candidates = list(pool.map(evaluate, transforms))
    else:
        regular_candidates = [evaluate(transform) for transform in transforms]

    # Always consider the original v5 winner. Signed context search is limited
    # to the most promising transform(s) to avoid an excessive encode-time hit.
    regular_candidates.sort(key=lambda x: x[0])
    candidates = list(regular_candidates)
    signed_count = 1 if effort < 9 else min(3, len(regular_candidates))
    for regular in regular_candidates[:signed_count]:
        filtered = regular[4]
        row_size = width * channels + channels
        residual = np.frombuffer(filtered, dtype=np.uint8).reshape(height, row_size)[:, channels:]
        small = np.count_nonzero((residual == 0) | (residual == 1) | (residual == 255))
        # This context model is designed for smooth/predictable residual fields.
        # Skip the extra entropy pass on texture/noise-like tiles where those
        # residual values are rare; the original v5 representation remains a
        # candidate and therefore file size cannot regress.
        if small / max(1, residual.size) < 0.08:
            continue
        signed_ids = _context2d_signed_ids(filtered, width, height, channels)
        signed_encoded = _rans_context2d_encode(
            filtered, width, height, channels, signed_ids)
        signed_payload = struct.pack("<I", len(filtered)) + signed_encoded
        candidates.append((len(signed_payload), CODEC_PRED2D_SIGNED,
                           regular[2], signed_payload, filtered))

    # Maximum effort additionally tests a 65-model left/up/upper-left context.
    # Its table cost is higher, so test only the best standard transform and
    # only when a substantial fraction of residuals are in the small classes.
    if effort >= 9 and regular_candidates:
        regular = regular_candidates[0]
        filtered = regular[4]
        row_size = width * channels + channels
        residual = np.frombuffer(filtered, dtype=np.uint8).reshape(height, row_size)[:, channels:]
        small = np.count_nonzero((residual == 0) | (residual == 1) | (residual == 255))
        if small / max(1, residual.size) >= 0.50:
            triple_ids = _context2d_triple_ids(filtered, width, height, channels)
            triple_encoded = _rans_context2d_encode(
                filtered, width, height, channels, triple_ids, context_count=65)
            triple_payload = struct.pack("<I", len(filtered)) + triple_encoded
            candidates.append((len(triple_payload), CODEC_PRED2D_TRIPLE,
                               regular[2], triple_payload, filtered))

    # Maximum effort additionally tests a context-mixing model. It blends
    # local 2D, per-channel and global symbol statistics rather than relying
    # on one context table in isolation.
    if effort >= 9 and regular_candidates:
        regular = regular_candidates[0]
        filtered = regular[4]
        row_size = width * channels + channels
        residual = np.frombuffer(filtered, dtype=np.uint8).reshape(height, row_size)[:, channels:]
        small = np.count_nonzero((residual == 0) | (residual == 1) | (residual == 255))
        if small / max(1, residual.size) >= 0.15:
            mixed = _rans_context2d_mixed_encode(filtered, width, height, channels)
            mixed_payload = struct.pack("<I", len(filtered)) + mixed
            candidates.append((len(mixed_payload), CODEC_PRED2D_MIXED,
                               regular[2], mixed_payload, filtered))

    best = min(candidates, key=lambda x: x[0])
    return best[1], best[2], best[3]


def _predictor_worker(job):
    raw, width, height, channels, effort = job
    return _predictor_candidate(raw, width, height, channels, effort, "cpu", parallel_transforms=False)


def _encode_rans_tokens(tokens: bytes, executor=None, effort="balanced") -> bytes:
    """Encode a token stream using PXL's context16 rANS.

    PXL v5 can optionally split large token streams into independent entropy
    chunks. It only keeps this representation when the complete chunked form
    is smaller than one continuous stream.
    """
    if not tokens:
        return b"\x00\x00\x00\x00" + core._rans_context16_encode(b"\x00")
    plain = struct.pack("<I", len(tokens)) + core._rans_context16_encode(tokens)
    try:
        from pxl_native import get_backend as _get_native_backend
        native_available = _get_native_backend(False) is not None
    except Exception:
        native_available = False
    if not native_available or len(tokens) < 131072:
        return plain
    chunk_size = {"fast": 131072, "balanced": 65536, "maximum": 32768}.get(effort, 65536)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    if len(chunks) < 2:
        return plain
    own_executor = None
    if executor is None:
        own_executor = ThreadPoolExecutor(max_workers=min(4, len(chunks)))
        executor = own_executor
    try:
        encoded_chunks = list(executor.map(core._rans_context16_encode, chunks))
    finally:
        if own_executor is not None:
            own_executor.shutdown(wait=True)
    packed = bytearray(struct.pack("<I", len(tokens)))
    packed.extend(b"PXMS")
    packed.extend(struct.pack("<II", chunk_size, len(chunks)))
    for raw_chunk, encoded in zip(chunks, encoded_chunks):
        packed.extend(struct.pack("<II", len(raw_chunk), len(encoded)))
        packed.extend(encoded)
    return bytes(packed) if len(packed) < len(plain) else plain


def _decode_rans_tokens(payload: bytes, allow_parallel=False, executor=None) -> bytes:
    if len(payload) < 4:
        raise ValueError("Truncated PXL token payload")
    size = int.from_bytes(payload[:4], "little")
    if size < 1 or size > MAX_TOKEN_STREAM:
        raise ValueError("Invalid PXL token stream size")
    if allow_parallel and payload[4:8] == b"PXMS":
        if len(payload) < 16:
            raise ValueError("Truncated PXL v5 parallel entropy header")
        chunk_size, count = struct.unpack_from("<II", payload, 8)
        if chunk_size < 1024 or chunk_size > 1048576 or count < 2 or count > 4096:
            raise ValueError("Invalid PXL v5 parallel entropy geometry")
        pos = 16
        jobs = []
        total_raw = 0
        for _ in range(count):
            if pos + 8 > len(payload):
                raise ValueError("Truncated PXL v5 entropy chunk header")
            raw_n, packed_n = struct.unpack_from("<II", payload, pos)
            pos += 8
            if raw_n < 1 or raw_n > chunk_size or packed_n < 5 or pos + packed_n > len(payload):
                raise ValueError("Invalid PXL v5 entropy chunk length")
            jobs.append((payload[pos:pos + packed_n], raw_n))
            pos += packed_n
            total_raw += raw_n
            if total_raw > size:
                raise ValueError("PXL v5 entropy chunks exceed declared token size")
        if pos != len(payload) or total_raw != size:
            raise ValueError("PXL v5 entropy chunk stream size mismatch")
        own_executor = None
        if executor is None:
            own_executor = ThreadPoolExecutor(max_workers=min(4, count))
            executor = own_executor
        try:
            parts = list(executor.map(lambda item: core._rans_context16_decode(item[0], item[1]), jobs))
        finally:
            if own_executor is not None:
                own_executor.shutdown(wait=True)
        return b"".join(parts)
    if len(payload) == 4:
        raise ValueError("Truncated token entropy stream")
    return core._rans_context16_decode(payload[4:], size)


def _extract_tile(raw_image: bytes, full_stride: int, tx: int, ty: int,
                  tw: int, th: int, channels: int) -> bytes:
    out = bytearray(tw * th * channels)
    p = 0
    for y in range(th):
        start = (ty + y) * full_stride + tx * channels
        n = tw * channels
        out[p:p + n] = raw_image[start:start + n]
        p += n
    return bytes(out)


def _tile_activity(raw: bytes, width: int, height: int, channels: int) -> float:
    """Mean local gradient magnitude; used only to choose reversible tile bounds."""
    a = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, channels).astype(np.int16)
    gx = np.abs(a[:, 1:, :] - a[:, :-1, :]).mean() if width > 1 else 0.0
    gy = np.abs(a[1:, :, :] - a[:-1, :, :]).mean() if height > 1 else 0.0
    return float((gx + gy) * 0.5)


def _adaptive_tile_regions(raw_image: bytes, width: int, height: int, channels: int,
                           tile_size: int, enabled: bool = True, max_depth: int = 2):
    """Partition the image into an explicit, non-overlapping variable tile layout.

    Splits only where subregion activity differs materially. Every internal
    boundary is aligned to eight pixels so PXL's transform and block paths keep
    their expected geometry. Image-edge tiles may be smaller/non-aligned.
    """
    stride = width * channels
    regions = []

    def extract(tx, ty, tw, th):
        out = bytearray(tw * th * channels)
        p = 0
        for yy in range(th):
            start = (ty + yy) * stride + tx * channels
            n = tw * channels
            out[p:p+n] = raw_image[start:start+n]
            p += n
        return bytes(out)

    def recurse(tx, ty, tw, th, depth):
        if not enabled or depth >= max_depth or min(tw, th) < 128:
            regions.append((tx, ty, tw, th))
            return
        # Split the longer side if the other side is already too narrow.
        boxes = []
        if tw >= 128 and th >= 128:
            left = ((tw // 2) // 8) * 8
            top = ((th // 2) // 8) * 8
            if left >= 64 and tw - left >= 64 and top >= 64 and th - top >= 64:
                boxes = [(tx, ty, left, top), (tx + left, ty, tw-left, top),
                         (tx, ty+top, left, th-top), (tx+left, ty+top, tw-left, th-top)]
        if not boxes and tw >= 128:
            left = ((tw // 2) // 8) * 8
            if left >= 64 and tw-left >= 64:
                boxes = [(tx, ty, left, th), (tx+left, ty, tw-left, th)]
        if not boxes and th >= 128:
            top = ((th // 2) // 8) * 8
            if top >= 64 and th-top >= 64:
                boxes = [(tx, ty, tw, top), (tx, ty+top, tw, th-top)]
        if not boxes:
            regions.append((tx, ty, tw, th))
            return
        activities = [_tile_activity(extract(*box), box[2], box[3], channels) for box in boxes]
        lo, hi = min(activities), max(activities)
        mean = sum(activities) / len(activities)
        # Avoid subdivision when regions are similarly smooth or similarly noisy.
        heterogeneity = hi - lo
        if hi < 5.0 or heterogeneity < 4.0 or hi < max(1.0, lo) * 2.35:
            regions.append((tx, ty, tw, th))
            return
        for box in boxes:
            recurse(*box, depth + 1)

    for y in range(0, height, tile_size):
        th = min(tile_size, height - y)
        for x in range(0, width, tile_size):
            tw = min(tile_size, width - x)
            recurse(x, y, tw, th, 0)
    regions.sort(key=lambda v: (v[1], v[0], v[3], v[2]))
    return regions


def _validate_v8_partition(regions, width, height):
    """Sweep-line coverage check; rejects gaps and overlapping rectangles."""
    events = {}
    for idx, (x, y, w, h) in enumerate(regions):
        if x < 0 or y < 0 or w < 1 or h < 1 or x+w > width or y+h > height:
            raise ValueError("PXL v8 tile lies outside the image")
        events.setdefault(y, []).append((1, idx, x, x+w))
        events.setdefault(y+h, []).append((-1, idx, x, x+w))
    ys = sorted(events)
    active = {}
    for pos, y in enumerate(ys[:-1]):
        for action, idx, x0, x1 in events[y]:
            if action < 0:
                active.pop(idx, None)
        for action, idx, x0, x1 in events[y]:
            if action > 0:
                active[idx] = (x0, x1)
        y2 = ys[pos+1]
        if y2 <= y:
            continue
        intervals = sorted(active.values())
        cursor = 0
        for x0, x1 in intervals:
            if x0 != cursor or x1 <= x0:
                raise ValueError("PXL v8 tile layout contains a gap or overlap")
            cursor = x1
        if cursor != width:
            raise ValueError("PXL v8 tile layout does not cover the image width")
    if not ys or ys[0] != 0 or ys[-1] != height:
        raise ValueError("PXL v8 tile layout does not cover the full image height")


def _write_tile_v8(f, tx, ty, tw, th, codec, transform, payload, decoded_size):
    if len(payload) > core.MAX_TILE_PAYLOAD:
        raise ValueError("PXL v8 tile payload exceeds safety limit")
    # Every v8 tile candidate competes against the verbatim pixel payload.
    # Keep this encoder invariant explicit so the decoder can safely enforce it.
    if len(payload) > decoded_size:
        raise ValueError("PXL v8 tile payload exceeds decoded tile size")
    crc = binascii.crc32(payload) & 0xFFFFFFFF
    f.write(V8_TILE_HEADER.pack(tx, ty, tw, th, codec, transform,
                                len(payload), decoded_size, crc))
    f.write(payload)


def _write_tile(f, tw, th, codec, transform, payload, decoded_size):
    if len(payload) > core.MAX_TILE_PAYLOAD:
        raise ValueError("PXL v4 tile payload exceeds safety limit")
    crc = binascii.crc32(payload) & 0xFFFFFFFF
    f.write(TILE_HEADER.pack(tw, th, codec, transform, len(payload), decoded_size, crc))
    f.write(payload)


def compress(input_path, output_path, mode="lossless", quality=88, tile_size=256,
             progress: Progress = None, effort="balanced", backend="auto", workers=None,
             adaptive_tiles=True, checkpoint_interval=V8_GROUP_INTERVAL):
    src, dst = Path(input_path), Path(output_path)
    if mode not in ("lossless", "lossy"):
        raise ValueError("mode must be lossless or lossy")
    levels = {"fast": 1, "balanced": 6, "maximum": 9}
    if effort not in levels:
        raise ValueError("effort must be fast, balanced, or maximum")
    if tile_size < core.MIN_TILE or tile_size > core.MAX_TILE or tile_size % 8:
        raise ValueError("tile_size must be a multiple of 8 between 64 and 1024")
    if mode == "lossy" and not 1 <= int(quality) <= 100:
        raise ValueError("quality must be between 1 and 100")
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("backend must be auto, cpu or cuda")
    requested_backend = backend
    if backend == "auto":
        backend = "cuda" if core._get_cupy(False) is not None else "cpu"
    if backend == "cuda":
        core._get_cupy(required=True)
        worker_count = 1
    else:
        count = os.cpu_count() or 1
        worker_count = count if workers == -1 else (min(8, max(1, count - 1)) if workers in (None, 0) else int(workers))
        if worker_count < 1 or worker_count > count:
            raise ValueError(f"workers must be between 1 and {count}, or -1 for all cores")
    # PXL v7 high-precision container: preserve uint16/float16/float32 samples
    # without normalizing them through Pillow's 8-bit RGB path.
    from pxl7_hdr import load_high_precision, compress_high_precision
    high_precision = load_high_precision(src)
    if high_precision is not None:
        if mode != "lossless":
            raise ValueError("PXL v7 high-precision images currently support lossless mode only; no HDR precision is discarded")
        # Preserve "auto" so each embedded byte-plane encode can fall back to
        # CPU if its first real predictor kernel fails after the probe.
        hp_backend = "auto" if requested_backend == "auto" else backend
        return compress_high_precision(src, dst, high_precision, tile_size, progress,
                                       effort, hp_backend, workers)
    with Image.open(src) as opened:
        img = core._normalise_image(opened)
    width, height = img.size
    channels = len(img.getbands())
    if width * height > core.MAX_PIXELS:
        raise ValueError("Image exceeds PXL's 100-megapixel safety limit")
    raw_image = img.tobytes()
    profile = PROFILE_LOSSLESS if mode == "lossless" else PROFILE_LOSSY
    stored_quality = 0 if mode == "lossless" else int(quality)
    if not isinstance(checkpoint_interval, int) or checkpoint_interval < 1 or checkpoint_interval > 256:
        raise ValueError("checkpoint_interval must be an integer between 1 and 256")
    full_stride = width * channels
    tile_regions = _adaptive_tile_regions(raw_image, width, height, channels, tile_size,
                                          enabled=bool(adaptive_tiles))
    tile_count = len(tile_regions)
    _validate_v8_partition(tile_regions, width, height)
    worker_count = min(worker_count, tile_count)
    px_state = PixelLZState(channels, tile_size)
    block_state = BlockReferenceState(channels)
    variable_block_state = VariableBlockReferenceState(channels)
    completed = 0
    cuda_tiles_used = 0
    cuda_fallback_reason = None
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Prediction analysis may run in independent CPU processes. Global reference
    # state is committed serially in tile order to keep the bitstream deterministic.
    jobs_enabled = backend == "cpu" and worker_count > 1 and tile_count > 1
    pool = None
    pending = deque()
    max_pending = max(2, worker_count * 2)
    try:
        from pxl_native import get_backend as _get_native_backend
        native_available = _get_native_backend(False) is not None
    except Exception:
        native_available = False
    entropy_executor = ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) if native_available else None

    def tile_jobs():
        for x, y, tile_w, tile_h in tile_regions:
            tile_raw = _extract_tile(raw_image, full_stride, x, y, tile_w, tile_h, channels)
            flat = _is_flat_tile(tile_raw, channels)
            high_entropy, repeated_triplets = _tile_traits(tile_raw, tile_w, tile_h, channels)
            yield (x, y, tile_w, tile_h, tile_raw, high_entropy, repeated_triplets, flat)

    tile_iter = iter(tile_jobs())

    def submit_more():
        while pool is not None and len(pending) < max_pending:
            try:
                x, y, tw, th, tile_raw, high_entropy, repeated_triplets, flat = next(tile_iter)
            except StopIteration:
                break
            future = None if (high_entropy or flat) else pool.submit(
                _predictor_worker, (tile_raw, tw, th, channels, levels[effort]))
            pending.append((future, x, y, tw, th, tile_raw, high_entropy, repeated_triplets, flat))

    try:
        if jobs_enabled:
            pool = ProcessPoolExecutor(max_workers=worker_count,
                                       mp_context=multiprocessing.get_context("spawn"))
            submit_more()
        with dst.open("wb") as f:
            f.write(HEADER.pack(MAGIC, VERSION, profile, channels, 0,
                                width, height, tile_size, stored_quality, 0))
            while True:
                if pool is not None:
                    if not pending:
                        break
                    future, tx, ty, tw, th, raw, high_entropy, repeated_triplets, flat = pending.popleft()
                    predictor = future.result() if future is not None else None
                    submit_more()
                else:
                    try:
                        tx, ty, tw, th, raw, high_entropy, repeated_triplets, flat = next(tile_iter)
                    except StopIteration:
                        break
                    predictor = None
                    if not (high_entropy or flat):
                        try:
                            predictor = _predictor_candidate(
                                raw, tw, th, channels, levels[effort], backend)
                            if backend == "cuda":
                                cuda_tiles_used += 1
                        except Exception as exc:
                            # Auto means best available backend, not "fail the
                            # whole encode if the GPU fails on a later kernel".
                            # Explicit CUDA remains strict so users can test it.
                            if requested_backend != "auto" or backend != "cuda":
                                raise
                            cuda_fallback_reason = f"{type(exc).__name__}: {exc}"
                            backend = "cpu"
                            worker_count = 1  # this stream is already running serially
                            predictor = _predictor_candidate(
                                raw, tw, th, channels, levels[effort], "cpu")
                            if progress:
                                progress(int(completed * 100 / max(1, tile_count)),
                                         "CUDA predictor failed; continuing with CPU: " + cuda_fallback_reason)

                group_start = (completed % checkpoint_interval) == 0
                if group_start and completed:
                    # Each v8 checkpoint begins an independent reference-history
                    # segment. Groups can therefore be reconstructed concurrently.
                    px_state = PixelLZState(channels, tile_size)
                    block_state = BlockReferenceState(channels)
                    variable_block_state = VariableBlockReferenceState(channels)

                if flat:
                    # Constant-color tiles are best represented by the palette
                    # path; don't run entropy searches or populate per-pixel hashes.
                    px_state.update(raw, index=False)
                    block_state.update(raw, tw, th)
                    variable_block_state.update(raw, tw, th)
                    palette = core._pack_palette(raw, channels)
                    candidates = [(len(raw), CODEC_RAW_PIXELS, 0, raw)]
                    if palette is not None:
                        candidates.append((len(palette), CODEC_PALETTE, 0, palette))
                    # A flat tile can still be smaller in the existing PXL-native
                    # residual+rANS representation (notably for tiny tiles where
                    # palette headers dominate). Compare the completed payload,
                    # while keeping the pixel-history skip flag deterministic.
                    legacy_codec, legacy_transform, legacy_payload = core._encode_lossless_tile(
                        raw, tw, th, channels, levels[effort], "cpu")
                    candidates.append((len(legacy_payload), legacy_codec, legacy_transform, legacy_payload))
                    _, codec, transform, payload = min(candidates, key=lambda x: x[0])
                    transform |= 0x80
                elif mode == "lossless":
                    # The 2D dictionary is updated for every tile. The pixel
                    # history may be advanced without indexing on incompressible
                    # tiles; v4 signals this so the decoder stays in sync.
                    candidates = [(len(raw), CODEC_RAW_PIXELS, 0, raw)]
                    skip_pixel_index = high_entropy and not repeated_triplets
                    if high_entropy and not repeated_triplets:
                        # An incompressible tile still seeds the block dictionary,
                        # but does not waste time building pixel-level hash chains.
                        px_state.update(raw, index=False)
                        block_state.update(raw, tw, th)
                        variable_block_state.update(raw, tw, th)
                    else:
                        block_tokens = block_state.encode(raw, tw, th)
                        block_payload = bytes((BLOCK_SIZE,)) + _encode_rans_tokens(block_tokens, entropy_executor, effort)
                        candidates.append((len(block_payload), CODEC_BLOCK2D, 0, block_payload))
                        pixel_tokens = px_state.encode(raw)
                        pixel_payload = _encode_rans_tokens(pixel_tokens, entropy_executor, effort)
                        candidates.append((len(pixel_payload), CODEC_PIXEL_LZ, 0, pixel_payload))
                        variable_block_trial = None
                        if (effort in ("balanced", "maximum") and repeated_triplets
                                and variable_block_state.has_large_matches(raw, tw, th)):
                            variable_block_trial = copy.deepcopy(variable_block_state)
                            variable_tokens = variable_block_trial.encode(raw, tw, th)
                            variable_payload = _encode_rans_tokens(variable_tokens, entropy_executor, effort)
                            candidates.append((len(variable_payload), CODEC_BLOCK2D_VAR, 0, variable_payload))
                    if not high_entropy:
                        legacy_codec, legacy_transform, legacy_payload = core._encode_lossless_tile(
                            raw, tw, th, channels, levels[effort], "cpu")
                        candidates.append((len(legacy_payload), legacy_codec, legacy_transform, legacy_payload))
                        if predictor is not None:
                            candidates.append((len(predictor[2]), predictor[0], predictor[1], predictor[2]))
                        palette = core._pack_palette(raw, channels)
                        if palette is not None:
                            candidates.append((len(palette), CODEC_PALETTE, 0, palette))
                    _, codec, transform, payload = min(candidates, key=lambda x: x[0])
                    if codec == CODEC_BLOCK2D_VAR and variable_block_trial is not None:
                        variable_block_state = variable_block_trial
                    elif not (high_entropy and not repeated_triplets):
                        variable_block_state.update(raw, tw, th)
                    if skip_pixel_index:
                        transform |= 0x80
                else:
                    # Lossy mode still compares exact representations for tiles
                    # where DCT would be wasteful. Trial dictionaries are cloned;
                    # only the chosen tile's reconstructed pixels are committed.
                    px_trial = copy.deepcopy(px_state)
                    block_trial = copy.deepcopy(block_state)
                    pixel_tokens = px_trial.encode(raw)
                    pixel_payload = _encode_rans_tokens(pixel_tokens, entropy_executor, effort)
                    block_tokens = block_trial.encode(raw, tw, th)
                    block_payload = bytes((BLOCK_SIZE,)) + _encode_rans_tokens(block_tokens, entropy_executor, effort)
                    lossy = core._encode_lossy_tile(raw, tw, th, channels,
                                                    stored_quality, levels[effort], backend)
                    legacy_codec, legacy_transform, legacy_payload = core._encode_lossless_tile(
                        raw, tw, th, channels, levels[effort], "cpu")
                    candidates = [(len(lossy), CODEC_DCT, 0, lossy),
                                  (len(raw), CODEC_RAW_PIXELS, 0, raw),
                                  (len(pixel_payload), CODEC_PIXEL_LZ, 0, pixel_payload),
                                  (len(block_payload), CODEC_BLOCK2D, 0, block_payload),
                                  (len(legacy_payload), legacy_codec, legacy_transform, legacy_payload)]
                    if predictor is not None:
                        candidates.append((len(predictor[2]), predictor[0], predictor[1], predictor[2]))
                    palette = core._pack_palette(raw, channels)
                    if palette is not None:
                        candidates.append((len(palette), CODEC_PALETTE, 0, palette))
                    _, codec, transform, payload = min(candidates, key=lambda x: x[0])
                    if codec == CODEC_DCT:
                        state_raw = core._decode_lossy_tile(lossy, tw, th, channels,
                                                            stored_quality, backend="cpu", version=3)
                    else:
                        state_raw = raw
                    px_state.update(state_raw)
                    block_state.update(state_raw, tw, th)

                if group_start:
                    transform |= 0x40  # v8 independent reference-history group
                _write_tile_v8(f, tx, ty, tw, th, codec, transform, payload, tw * th * channels)
                completed += 1
                if backend == "cuda" and cuda_tiles_used:
                    accelerator = f"CUDA predictor used on {cuda_tiles_used} tile(s); entropy/dictionaries on CPU"
                elif backend == "cuda":
                    accelerator = "CUDA available; no predictor-suitable tile yet (other tile paths are CPU-side)"
                else:
                    accelerator = f"CPU {worker_count} worker(s)"
                if progress:
                    progress(int(completed * 100 / tile_count),
                             f"PXL v8 encoded tile {completed}/{tile_count} · {accelerator}")
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if entropy_executor is not None:
            entropy_executor.shutdown(wait=True)

    output_bytes = dst.stat().st_size
    return {"input_bytes": src.stat().st_size, "output_bytes": output_bytes,
            "width": width, "height": height, "mode": mode, "quality": stored_quality,
            "tiles": tile_count, "backend": backend, "workers": worker_count,
            "cuda_tiles_used": cuda_tiles_used,
            "cuda_fallback": cuda_fallback_reason,
            "version": VERSION, "adaptive_tiles": bool(adaptive_tiles),
            "checkpoint_interval": checkpoint_interval, "tile_layout": "adaptive" if adaptive_tiles else "fixed",
            "ratio": output_bytes / max(1, src.stat().st_size)}


def _decode_pred(payload: bytes, width: int, height: int, channels: int, transform: int,
                 signed_context: bool = False, triple_context: bool = False) -> bytes:
    if len(payload) < 5:
        raise ValueError("Truncated PXL v4 predictor payload")
    expected = int.from_bytes(payload[:4], "little")
    expected_geometry = height * (width * channels + channels)
    if expected != expected_geometry:
        raise ValueError("Invalid PXL v4 predictor decoded size")
    filtered = _rans_context2d_decode(payload[4:], expected, width, height, channels,
                                      signed_context=signed_context,
                                      triple_context=triple_context)
    try:
        from pxl_native import get_backend as _get_native_backend
        native = _get_native_backend(False)
    except Exception:
        native = None
    if native is not None:
        return native.unfilter_color_v3(filtered, width, height, channels, transform)
    restored = _unfilter_rows_v3_fast(filtered, width, height, channels)
    return core._inverse_color(restored, channels, transform)


def _decode_v8_group(entries, channels, profile, quality, tile_size):
    """Decode one independent reference-history group in raster stream order."""
    px_state = PixelLZState(channels, tile_size)
    block_state = BlockReferenceState(channels)
    variable_block_state = VariableBlockReferenceState(channels)
    decoded = []
    for rec in entries:
        tx, ty, tw, th, codec, transform_flags, payload, decoded_size = rec
        skip_pixel_index = bool(transform_flags & 0x80)
        actual_transform = transform_flags & 0x3F
        if codec == CODEC_RAW_PIXELS:
            if actual_transform != 0 or len(payload) != decoded_size:
                raise ValueError("Invalid PXL v8 raw tile")
            raw = payload
        elif codec == CODEC_PRED2D:
            if skip_pixel_index:
                raise ValueError("Invalid pixel-history flag on PXL predictor tile")
            raw = _decode_pred(payload, tw, th, channels, actual_transform)
        elif codec == CODEC_PRED2D_SIGNED:
            if skip_pixel_index:
                raise ValueError("Invalid pixel-history flag on signed-context predictor tile")
            raw = _decode_pred(payload, tw, th, channels, actual_transform, signed_context=True)
        elif codec == CODEC_PRED2D_TRIPLE:
            if skip_pixel_index:
                raise ValueError("Invalid pixel-history flag on triple-context predictor tile")
            raw = _decode_pred(payload, tw, th, channels, actual_transform, triple_context=True)
        elif codec == CODEC_PRED2D_MIXED:
            if skip_pixel_index or len(payload) < 5:
                raise ValueError("Invalid mixed-context predictor flags or payload")
            expected = int.from_bytes(payload[:4], "little")
            if expected != th * (tw * channels + channels):
                raise ValueError("Invalid PXL mixed-context decoded size")
            filtered = _rans_context2d_mixed_decode(payload[4:], expected, tw, th, channels)
            raw = core._inverse_color(_unfilter_rows_v3_fast(filtered, tw, th, channels),
                                      channels, actual_transform)
        elif codec == CODEC_BLOCK2D_VAR:
            if actual_transform != 0:
                raise ValueError("Variable block references cannot use a colour transform")
            tokens = _decode_rans_tokens(payload, allow_parallel=True)
            raw = variable_block_state.decode(tokens, tw, th)
        elif codec == CODEC_PIXEL_LZ:
            if actual_transform != 0:
                raise ValueError("Invalid PXL pixel-LZ transform")
            tokens = _decode_rans_tokens(payload, allow_parallel=True)
            raw = px_state.decode(tokens, tw * th)
        elif codec == CODEC_BLOCK2D:
            if actual_transform != 0 or len(payload) < 5 or payload[0] != BLOCK_SIZE:
                raise ValueError("Invalid PXL block-reference header")
            tokens = _decode_rans_tokens(payload[1:], allow_parallel=True)
            raw = block_state.decode(tokens, tw, th)
        elif codec == CODEC_PALETTE:
            if actual_transform != 0:
                raise ValueError("Invalid PXL palette transform")
            raw = core._unpack_palette(payload, tw, th, channels)
        elif codec == CODEC_DCT and profile == PROFILE_LOSSY:
            if actual_transform != 0:
                raise ValueError("Invalid PXL DCT transform")
            raw = core._decode_lossy_tile(payload, tw, th, channels, quality, backend="cpu", version=3)
        else:
            if profile == PROFILE_LOSSLESS and codec == CODEC_DCT:
                raise ValueError("DCT tile in lossless PXL file")
            raw = core._decode_lossless_tile(payload, tw, th, channels, codec, actual_transform, version=3)
        if len(raw) != decoded_size:
            raise ValueError("Decoded PXL v8 tile size mismatch")
        # These transitions deliberately mirror the v4-v7 encoder's committed
        # state semantics. The first tile of every group starts from empty state.
        if codec != CODEC_PIXEL_LZ:
            # Non-PixelLZ candidates still advance the encoder's pixel index
            # unless the explicit skip-index flag was stored in the tile.
            px_state.update(raw, index=not skip_pixel_index)
        if codec != CODEC_BLOCK2D:
            block_state.update(raw, tw, th)
        if codec != CODEC_BLOCK2D_VAR:
            variable_block_state.update(raw, tw, th)
        decoded.append((tx, ty, tw, th, raw))
    return decoded


def _decompress_v8_stream(stream, src, dst, profile, channels, flags, width, height,
                          tile_size, quality, progress=None, workers=None):
    if flags != 0:
        raise ValueError("Unsupported PXL v8 image header flags")
    output = bytearray(width * height * channels)
    groups = []
    current = None
    regions = []
    count = 0
    payload_total = 0
    current_group_tiles = 0
    image_bytes = width * height * channels
    while True:
        header = stream.read(V8_TILE_HEADER.size)
        if not header:
            break
        if len(header) != V8_TILE_HEADER.size:
            raise ValueError("Truncated PXL v8 tile header")
        tx, ty, tw, th, codec, transform, payload_size, decoded_size, crc = V8_TILE_HEADER.unpack(header)
        count += 1
        if count > 100000:
            raise ValueError("PXL v8 tile count exceeds safety limit")
        if tw < 1 or th < 1 or tw > tile_size or th > tile_size or tx % 8 or ty % 8:
            raise ValueError("Invalid PXL v8 tile geometry")
        if tx + tw > width or ty + th > height or decoded_size != tw * th * channels:
            raise ValueError("PXL v8 tile lies outside image or has invalid decoded size")
        if transform & 0x3F > 4:
            raise ValueError("Unknown PXL v8 tile transform")
        if payload_size > core.MAX_TILE_PAYLOAD:
            raise ValueError("PXL v8 tile exceeds payload safety limit")
        # The encoder always evaluates the raw tile as a candidate, so no
        # valid v8 tile can be larger than its decoded pixels. Enforcing this
        # also bounds aggregate retained compressed data by the image size.
        if payload_size > decoded_size:
            raise ValueError("PXL v8 tile payload exceeds decoded tile size")
        if payload_size > image_bytes - payload_total:
            raise ValueError("PXL v8 aggregate tile payload exceeds image-size limit")
        payload_total += payload_size
        payload = stream.read(payload_size)
        if len(payload) != payload_size:
            raise ValueError("Truncated PXL v8 tile payload")
        if (binascii.crc32(payload) & 0xFFFFFFFF) != crc:
            raise ValueError("PXL v8 tile payload checksum failed")
        if transform & 0x40:
            current = []
            groups.append(current)
            current_group_tiles = 0
        elif current is None:
            raise ValueError("First PXL v8 tile must begin an independent reference group")
        current_group_tiles += 1
        if current_group_tiles > V8_GROUP_MAX_TILES:
            raise ValueError("PXL v8 reference group exceeds the 256-tile safety limit")
        current.append((tx, ty, tw, th, codec, transform, payload, decoded_size))
        regions.append((tx, ty, tw, th))
    if not count or not groups:
        raise ValueError("PXL v8 file contains no tiles")
    _validate_v8_partition(regions, width, height)

    cpu_count = os.cpu_count() or 1
    decode_workers = cpu_count if workers == -1 else (min(4, cpu_count) if workers in (None, 0) else max(1, int(workers)))
    decode_workers = max(1, min(decode_workers, len(groups)))
    completed = 0
    if decode_workers > 1 and len(groups) > 1:
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            results = pool.map(lambda g: _decode_v8_group(g, channels, profile, quality, tile_size), groups)
            for group_result in results:
                for tx, ty, tw, th, raw in group_result:
                    p = 0
                    for y in range(th):
                        dst_pos = (ty + y) * (width * channels) + tx * channels
                        n = tw * channels
                        output[dst_pos:dst_pos+n] = raw[p:p+n]
                        p += n
                    completed += 1
                    if progress:
                        progress(int(completed * 100 / count),
                                 f"PXL v8 decoded tile {completed}/{count} · {decode_workers} independent groups")
    else:
        for group in groups:
            for tx, ty, tw, th, raw in _decode_v8_group(group, channels, profile, quality, tile_size):
                p = 0
                for y in range(th):
                    dst_pos = (ty + y) * (width * channels) + tx * channels
                    n = tw * channels
                    output[dst_pos:dst_pos+n] = raw[p:p+n]
                    p += n
                completed += 1
                if progress:
                    progress(int(completed * 100 / count), f"PXL v8 decoded tile {completed}/{count}")
    mode = "L" if channels == 1 else ("RGB" if channels == 3 else "RGBA")
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes(mode, (width, height), bytes(output)).save(dst)
    return {"output_bytes": dst.stat().st_size, "width": width, "height": height,
            "mode": "lossless" if profile == PROFILE_LOSSLESS else "lossy",
            "quality": quality, "tiles": count, "backend": "cpu",
            "workers": decode_workers, "parallel_groups": len(groups), "version": 8}


def decompress(input_path, output_path, progress: Progress = None, backend="auto", workers=None):
    src, dst = Path(input_path), Path(output_path)
    with src.open("rb") as f:
        head = f.read(HEADER.size)
        if len(head) != HEADER.size:
            raise ValueError("Truncated PXL v4 header")
        magic, version, profile, channels, flags, width, height, tile_size, quality, reserved = HEADER.unpack(head)
        if magic != MAGIC or version not in (4, 5, 6, 7, 8):
            raise ValueError("Not a supported PXL v4/v5/v6/v7/v8 file")
        core._validate_header(3, profile, channels, width, height, tile_size, quality)
        if version == 8:
            if reserved != 0:
                raise ValueError("Unsupported PXL v8 reserved header byte")
            if backend not in ("auto", "cpu", "cuda"):
                raise ValueError("backend must be auto, cpu or cuda")
            return _decompress_v8_stream(f, src, dst, profile, channels, flags, width, height,
                                         tile_size, quality, progress=progress, workers=workers)
        if flags != 0 or reserved != 0:
            raise ValueError("Unsupported PXL v4 header flags")
        if backend not in ("auto", "cpu", "cuda"):
            raise ValueError("backend must be auto, cpu or cuda")
        # The cross-tile reference stream is intentionally sequential. CUDA is
        # not beneficial for decode while entropy/token expansion is serial.
        backend_used = "cpu"
        mode = "L" if channels == 1 else ("RGB" if channels == 3 else "RGBA")
        output = bytearray(width * height * channels)
        full_stride = width * channels
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)
        tile_count = tiles_x * tiles_y
        px_state = PixelLZState(channels, tile_size)
        block_state = BlockReferenceState(channels)
        variable_block_state = VariableBlockReferenceState(channels)
        completed = 0
        try:
            from pxl_native import get_backend as _get_native_backend
            native_available = _get_native_backend(False) is not None
        except Exception:
            native_available = False
        decode_workers = (os.cpu_count() or 1) if workers == -1 else (min(4, os.cpu_count() or 1) if workers in (None, 0) else max(1, int(workers)))
        entropy_executor = ThreadPoolExecutor(max_workers=max(1, min(decode_workers, 8))) if native_available else None
        for ty in range(0, height, tile_size):
            th = min(tile_size, height - ty)
            for tx in range(0, width, tile_size):
                tw = min(tile_size, width - tx)
                header = f.read(TILE_HEADER.size)
                if len(header) != TILE_HEADER.size:
                    raise ValueError("Unexpected end of PXL v4 tile header")
                tw2, th2, codec, transform, payload_size, decoded_size, crc = TILE_HEADER.unpack(header)
                if tw2 != tw or th2 != th or decoded_size != tw * th * channels:
                    raise ValueError("Invalid PXL v4 tile geometry")
                if payload_size > core.MAX_TILE_PAYLOAD:
                    raise ValueError("PXL v4 tile exceeds payload safety limit")
                payload = f.read(payload_size)
                if len(payload) != payload_size:
                    raise ValueError("Truncated PXL v4 tile payload")
                if (binascii.crc32(payload) & 0xFFFFFFFF) != crc:
                    raise ValueError("PXL v4 tile payload checksum failed")
                skip_pixel_index = bool(transform & 0x80)
                actual_transform = transform & 0x7F
                if codec == CODEC_RAW_PIXELS:
                    if actual_transform != 0 or len(payload) != decoded_size:
                        raise ValueError("Invalid PXL v4 raw tile")
                    raw = payload
                elif codec == CODEC_PRED2D:
                    if skip_pixel_index:
                        raise ValueError("Invalid pixel-history flag on PXL predictor tile")
                    raw = _decode_pred(payload, tw, th, channels, actual_transform)
                elif codec == CODEC_PRED2D_SIGNED:
                    if version < 6 or skip_pixel_index:
                        raise ValueError("Signed-context predictor is only valid in PXL v6/v7")
                    raw = _decode_pred(payload, tw, th, channels, actual_transform,
                                       signed_context=True)
                elif codec == CODEC_PRED2D_TRIPLE:
                    if version < 6 or skip_pixel_index:
                        raise ValueError("Triple-context predictor is only valid in PXL v6/v7")
                    raw = _decode_pred(payload, tw, th, channels, actual_transform,
                                       triple_context=True)
                elif codec == CODEC_PRED2D_MIXED:
                    if version < 7 or skip_pixel_index or len(payload) < 5:
                        raise ValueError("Mixed-context predictor is only valid in PXL v8")
                    expected = int.from_bytes(payload[:4], "little")
                    if expected != th * (tw * channels + channels):
                        raise ValueError("Invalid PXL mixed-context decoded size")
                    filtered = _rans_context2d_mixed_decode(payload[4:], expected, tw, th, channels)
                    raw = core._inverse_color(_unfilter_rows_v3_fast(filtered, tw, th, channels),
                                              channels, actual_transform)
                elif codec == CODEC_BLOCK2D_VAR:
                    if version < 7 or transform != 0:
                        raise ValueError("Variable block references are only valid in PXL v8")
                    tokens = _decode_rans_tokens(payload, allow_parallel=(version >= 5), executor=entropy_executor)
                    raw = variable_block_state.decode(tokens, tw, th)
                elif codec == CODEC_PIXEL_LZ:
                    if transform != 0:
                        raise ValueError("Invalid PXL v4 pixel-LZ transform")
                    tokens = _decode_rans_tokens(payload, allow_parallel=(version >= 5), executor=entropy_executor)
                    raw = px_state.decode(tokens, tw * th)
                elif codec == CODEC_BLOCK2D:
                    if actual_transform != 0 or len(payload) < 5 or payload[0] != BLOCK_SIZE:
                        raise ValueError("Invalid PXL v4 block-reference header")
                    tokens = _decode_rans_tokens(payload[1:], allow_parallel=(version >= 5), executor=entropy_executor)
                    raw = block_state.decode(tokens, tw, th)
                elif codec == CODEC_PALETTE:
                    if actual_transform != 0:
                        raise ValueError("Invalid PXL v4 palette transform")
                    raw = core._unpack_palette(payload, tw, th, channels)
                elif codec == CODEC_DCT and profile == PROFILE_LOSSY:
                    if actual_transform != 0:
                        raise ValueError("Invalid PXL v4 DCT transform")
                    raw = core._decode_lossy_tile(payload, tw, th, channels, quality, backend="cpu", version=3)
                else:
                    # v4 may use self-contained v3 codecs as measured fallbacks.
                    if profile == PROFILE_LOSSLESS and codec == CODEC_DCT:
                        raise ValueError("DCT tile in lossless PXL v4 file")
                    raw = core._decode_lossless_tile(payload, tw, th, channels, codec, actual_transform, version=3)

                if len(raw) != decoded_size:
                    raise ValueError("Decoded PXL v4 tile size mismatch")
                # Maintain global dictionaries in exactly tile traversal order.
                if codec != CODEC_PIXEL_LZ:
                    px_state.update(raw, index=False)
                if codec != CODEC_BLOCK2D:
                    block_state.update(raw, tw, th)
                if codec != CODEC_BLOCK2D_VAR:
                    variable_block_state.update(raw, tw, th)
                p = 0
                for y in range(th):
                    dst_pos = (ty + y) * full_stride + tx * channels
                    n = tw * channels
                    output[dst_pos:dst_pos + n] = raw[p:p + n]
                    p += n
                completed += 1
                if progress:
                    progress(int(completed * 100 / tile_count),
                             f"PXL v{version} decoded tile {completed}/{tile_count} · sequential reference stream")
        if f.read(1):
            raise ValueError("Trailing data after final PXL tile")
        if entropy_executor is not None:
            entropy_executor.shutdown(wait=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes(mode, (width, height), bytes(output)).save(dst)
    return {"output_bytes": dst.stat().st_size, "width": width, "height": height,
            "mode": "lossless" if profile == PROFILE_LOSSLESS else "lossy",
            "quality": quality, "tiles": tile_count, "backend": backend_used,
            "workers": 1, "version": version}


def inspect(path) -> dict:
    p = Path(path)
    with p.open("rb") as f:
        h = f.read(HEADER.size)
    if len(h) != HEADER.size:
        raise ValueError("Truncated PXL v4 header")
    magic, version, profile, channels, flags, width, height, tile_size, quality, reserved = HEADER.unpack(h)
    if magic != MAGIC or version not in (4, 5, 6, 7, 8):
        raise ValueError("Not a supported PXL v4/v5/v6/v7/v8 file")
    return {"version": version, "profile": "lossless" if profile == PROFILE_LOSSLESS else "lossy",
            "width": width, "height": height, "channels": channels,
            "tile_size": tile_size, "quality": quality,
            "file_size": p.stat().st_size, "format": f"PXL v{version} custom cross-tile/native entropy"}
