// MSVC-specific AVX2 byte equality kernel. Compiled separately with /arch:AVX2
// so the baseline library can safely dispatch only after CPUID detection.
#include <cstdint>
#include <cstddef>
#include <immintrin.h>
#include <intrin.h>

extern "C" uint64_t pxl_msvc_equal_avx2(const uint8_t* a, const uint8_t* b, uint32_t n) {
    if (!a || !b) return 0;
    size_t i = 0;
    uint64_t count = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const unsigned mask = static_cast<unsigned>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(x, y)));
        count += __popcnt(mask);
    }
    for (; i < n; ++i) count += (a[i] == b[i]);
    return count;
}
