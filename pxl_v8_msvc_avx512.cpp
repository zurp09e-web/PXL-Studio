// MSVC-specific AVX-512BW byte equality kernel. Compiled separately with
// /arch:AVX512; called only when CPUID and XCR0 confirm OS support.
#include <cstdint>
#include <cstddef>
#include <immintrin.h>
#include <intrin.h>

extern "C" uint64_t pxl_msvc_equal_avx512(const uint8_t* a, const uint8_t* b, uint32_t n) {
    if (!a || !b) return 0;
    size_t i = 0;
    uint64_t count = 0;
    for (; i + 64 <= n; i += 64) {
        const __m512i x = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        const __m512i y = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        const unsigned __int64 mask = static_cast<unsigned __int64>(_mm512_cmpeq_epi8_mask(x, y));
        count += __popcnt64(mask);
    }
    for (; i < n; ++i) count += (a[i] == b[i]);
    return count;
}
