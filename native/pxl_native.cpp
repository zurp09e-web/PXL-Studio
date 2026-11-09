// PXL native acceleration library — no third-party compression dependencies.
// ABI is intentionally C-compatible so the Python application can load it via ctypes.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <new>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <deque>

#if defined(_WIN32)
#  define PXL_EXPORT __declspec(dllexport)
#else
#  define PXL_EXPORT __attribute__((visibility("default")))
#endif

namespace {
constexpr uint32_t RANS_SCALE_BITS = 12;
constexpr uint32_t RANS_TOTAL = 1u << RANS_SCALE_BITS;
constexpr uint32_t RANS_L = 1u << 23;

inline uint8_t paeth(uint8_t a, uint8_t b, uint8_t c) {
    const int p = int(a) + int(b) - int(c);
    const int pa = std::abs(p - int(a));
    const int pb = std::abs(p - int(b));
    const int pc = std::abs(p - int(c));
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

inline int predictor(int id, uint8_t left, uint8_t up, uint8_t ul) {
    switch (id) {
        case 0: return 0;
        case 1: return left;
        case 2: return up;
        case 3: return (int(left) + int(up)) / 2;
        case 4: return paeth(left, up, ul);
        case 5: return int(left) + int(up) - int(ul);
        case 6: {
            const int lo = std::min(int(left), int(up));
            const int hi = std::max(int(left), int(up));
            if (int(ul) >= hi) return lo;
            if (int(ul) <= lo) return hi;
            return int(left) + int(up) - int(ul);
        }
        default: return 0;
    }
}

inline void forward_pixel(const uint8_t* s, uint8_t* d, int ch, int transform) {
    if (ch == 1 || transform == 0) {
        for (int c = 0; c < ch; ++c) d[c] = s[c];
        return;
    }
    const uint32_t r=s[0], g=s[1], b=s[2];
    switch (transform) {
        case 1: // G, R-G, B-G
            d[0]=uint8_t(g); d[1]=uint8_t(r-g); d[2]=uint8_t(b-g); break;
        case 2: // R, G-R, B-R
            d[0]=uint8_t(r); d[1]=uint8_t(g-r); d[2]=uint8_t(b-r); break;
        case 3: // B, R-B, G-B
            d[0]=uint8_t(b); d[1]=uint8_t(r-b); d[2]=uint8_t(g-b); break;
        case 4: { // reversible YCoCg-R variant used by PXL
            const uint32_t co = (r - b) & 255u;
            const uint32_t t = (b + (co >> 1)) & 255u;
            const uint32_t cg = (g - t) & 255u;
            const uint32_t y = (t + (cg >> 1)) & 255u;
            d[0]=uint8_t(y); d[1]=uint8_t(cg); d[2]=uint8_t(co); break;
        }
        default:
            for (int c=0;c<ch;++c) d[c]=s[c];
            return;
    }
    if (ch == 4) d[3] = s[3];
}

inline void inverse_pixel(const uint8_t* s, uint8_t* d, int ch, int transform) {
    if (ch == 1 || transform == 0) {
        for (int c = 0; c < ch; ++c) d[c] = s[c];
        return;
    }
    const uint32_t x=s[0], y=s[1], z=s[2];
    switch (transform) {
        case 1: d[0]=uint8_t(y+x); d[1]=uint8_t(x); d[2]=uint8_t(z+x); break;
        case 2: d[0]=uint8_t(x); d[1]=uint8_t(y+x); d[2]=uint8_t(z+x); break;
        case 3: d[0]=uint8_t(y+x); d[1]=uint8_t(z+x); d[2]=uint8_t(x); break;
        case 4: {
            const uint32_t t = (x - (y >> 1)) & 255u;
            const uint32_t g = (y + t) & 255u;
            const uint32_t b = (t - (z >> 1)) & 255u;
            const uint32_t r = (b + z) & 255u;
            d[0]=uint8_t(r); d[1]=uint8_t(g); d[2]=uint8_t(b); break;
        }
        default: for (int c=0;c<ch;++c) d[c]=s[c]; return;
    }
    if (ch == 4) d[3] = s[3];
}

inline uint8_t signed_residual_class(uint8_t value) {
    // Keep the three most common small prediction residuals distinct.
    if (value == 0) return 0;
    if (value == 1) return 1;
    if (value == 255) return 2;
    return 3;
}

inline uint8_t context2d(const uint8_t* out, uint32_t i, uint32_t row_size,
                         uint32_t width, uint32_t ch) {
    const uint32_t row_offset = i % row_size;
    if (row_offset < ch) return 16;
    const uint32_t sample_offset = row_offset - ch;
    const uint32_t x = sample_offset / ch;
    const uint8_t left = x ? out[i - ch] : 0;
    const uint8_t up = i >= row_size ? out[i - row_size] : 0;
    return uint8_t(((left >> 6) << 2) | (up >> 6));
}

int rans_encode_ids(const uint8_t* data, const uint8_t* ids, uint32_t n,
                    const uint16_t* freq, const uint16_t* cumul,
                    uint8_t* out, uint32_t cap, uint32_t* out_n, uint32_t* out_state) {
    if (!data || !ids || !freq || !cumul || !out || !out_n || !out_state) return -1;
    uint64_t state = RANS_L;
    std::vector<uint8_t> reverse;
    reverse.reserve(n / 4 + 16);
    for (uint32_t ri = n; ri > 0; --ri) {
        const uint32_t i = ri - 1;
        const uint32_t ctx = ids[i];
        const uint32_t sym = data[i];
        const uint32_t f = freq[ctx * 256u + sym];
        if (f == 0) return -2;
        const uint64_t x_max = ((uint64_t(RANS_L) >> RANS_SCALE_BITS) << 8) * f;
        while (state >= x_max) {
            reverse.push_back(uint8_t(state & 255u));
            state >>= 8;
        }
        state = ((state / f) << RANS_SCALE_BITS) + (state % f) + cumul[ctx * 256u + sym];
        if (state > 0xffffffffULL) return -3;
    }
    if (reverse.size() > cap) return -4;
    for (size_t i = 0; i < reverse.size(); ++i) out[i] = reverse[reverse.size() - 1 - i];
    *out_n = uint32_t(reverse.size());
    *out_state = uint32_t(state);
    return 0;
}

int rans_decode(const uint8_t* blob, uint32_t blob_n, uint32_t start,
                uint32_t output_n, int mode, uint32_t width, uint32_t height,
                uint32_t channels, uint32_t contexts,
                const uint16_t* freq, const uint16_t* cumul,
                const uint8_t* decode, uint8_t* out, uint32_t* end_pos) {
    if (!blob || !freq || !cumul || !decode || !out || !end_pos) return -1;
    if (start + 4 > blob_n || (mode != 16 && mode != 2 && mode != 3 && mode != 4 && mode != 5)) return -2;
    uint64_t state = uint64_t(blob[start]) | (uint64_t(blob[start+1]) << 8) |
                     (uint64_t(blob[start+2]) << 16) | (uint64_t(blob[start+3]) << 24);
    if (state < RANS_L) return -3;
    uint32_t pos = start + 4;
    const uint32_t row_size = width * channels + channels;
    for (uint32_t i = 0; i < output_n; ++i) {
        uint32_t ctx = 0;
        if (mode == 16) ctx = i ? (out[i-1] >> 4) : 0;
        else if (mode == 2 || mode == 3 || mode == 4) {
            const uint32_t row_offset = i % row_size;
            if (row_offset < channels) ctx = (mode == 4) ? 64 : 16;
            else {
                const uint32_t sample_offset = row_offset - channels;
                const uint32_t x = sample_offset / channels;
                const uint8_t left = x ? out[i - channels] : 0;
                const uint8_t up = i >= row_size ? out[i - row_size] : 0;
                if (mode == 2) ctx = uint8_t(((left >> 6) << 2) | (up >> 6));
                else if (mode == 3) ctx = uint8_t(signed_residual_class(left) * 4 + signed_residual_class(up));
                else {
                    const uint8_t upper_left = (x && i >= row_size) ? out[i - row_size - channels] : 0;
                    ctx = uint8_t(signed_residual_class(left) * 16 +
                                  signed_residual_class(up) * 4 +
                                  signed_residual_class(upper_left));
                }
            }
        } else {
            const uint32_t spatial_row = width * channels + 1;
            const uint32_t col = i % spatial_row;
            if (!col) ctx = channels * 4;
            else {
                const uint32_t sample_pos = col - 1;
                const uint32_t ch = sample_pos % channels;
                const uint8_t previous = sample_pos >= channels ? out[i-channels] : 0;
                ctx = ch * 4 + (previous >> 6);
            }
        }
        if (ctx >= contexts) return -4;
        const uint32_t slot = uint32_t(state & (RANS_TOTAL - 1));
        const uint32_t sym = decode[ctx * RANS_TOTAL + slot];
        const uint32_t f = freq[ctx * 256u + sym];
        if (!f) return -5;
        state = uint64_t(f) * (state >> RANS_SCALE_BITS) + slot - cumul[ctx * 256u + sym];
        while (state < RANS_L) {
            if (pos >= blob_n) return -6;
            state = (state << 8) | blob[pos++];
        }
        out[i] = uint8_t(sym);
    }
    *end_pos = pos;
    return 0;
}
inline uint64_t pixel_hash(const uint8_t* p, uint32_t channels) {
    uint64_t h = 1469598103934665603ULL;
    for (uint32_t i=0; i<3*channels; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h ? h : 1ULL;
}
inline uint32_t varint_size(uint64_t v) {
    uint32_t n=1;
    while (v>=128) { ++n; v >>= 7; }
    return n;
}
inline void write_varint(std::vector<uint8_t>& out, uint64_t v) {
    while (v>=128) { out.push_back(uint8_t((v&127)|128)); v >>= 7; }
    out.push_back(uint8_t(v));
}
inline bool read_varint(const uint8_t* data, uint32_t n, uint32_t& pos, uint64_t& value) {
    value=0; uint32_t shift=0;
    for (int i=0;i<10;++i) {
        if (pos>=n || shift>=64) return false;
        const uint8_t b=data[pos++];
        value |= uint64_t(b&127) << shift;
        if (!(b&128)) return true;
        shift += 7;
    }
    return false;
}

int pixel_lz_encode_impl(const uint8_t* history, uint32_t history_n,
                         const uint8_t* raw, uint32_t raw_n, uint32_t channels,
                         uint32_t window_pixels, uint8_t* out, uint32_t cap,
                         uint32_t* out_n) {
    if (!raw || !out || !out_n || channels<1 || channels>4 || raw_n%channels || history_n%channels) return -1;
    const uint32_t hist_pixels=history_n/channels;
    const uint32_t pixels=raw_n/channels;
    const uint32_t combined_n=history_n+raw_n;
    std::vector<uint8_t> combined(combined_n);
    if (history_n) std::memcpy(combined.data(),history,history_n);
    std::memcpy(combined.data()+history_n,raw,raw_n);
    std::unordered_map<uint64_t,std::vector<uint32_t>> heads;
    heads.reserve(std::min<uint32_t>(hist_pixels+pixels, 1u<<20));
    auto insert_pos=[&](uint32_t pos) {
        if (uint64_t(pos)+2 >= uint64_t(hist_pixels)+pixels) return;
        const uint64_t key=pixel_hash(combined.data()+uint64_t(pos)*channels,channels);
        auto& chain=heads[key];
        chain.push_back(pos);
        if (chain.size()>10) chain.erase(chain.begin());
    };
    std::vector<uint8_t> tokens;
    tokens.reserve(raw_n + pixels/8 + 64);
    std::vector<uint8_t> literals;
    uint32_t literal_count=0, literal_start=0, indexed_until=0;
    auto flush_literals=[&]() {
        if (!literal_count) return;
        tokens.push_back(0);
        write_varint(tokens,literal_count);
        const uint32_t start=history_n+literal_start*channels;
        const uint32_t n=literal_count*channels;
        tokens.insert(tokens.end(),combined.begin()+start,combined.begin()+start+n);
        literal_count=0;
    };
    uint32_t i=0;
    while (i<pixels) {
        const uint32_t target=hist_pixels+i;
        while (indexed_until<target) insert_pos(indexed_until++);
        const uint32_t local=history_n+i*channels;
        uint32_t best_len=0, best_dist=0;
        if (i+3<=pixels) {
            const uint64_t key=pixel_hash(combined.data()+local,channels);
            auto it=heads.find(key);
            if (it!=heads.end()) {
                const auto& chain=it->second;
                int checked=0;
                for (auto rit=chain.rbegin(); rit!=chain.rend() && checked<10; ++rit,++checked) {
                    const uint32_t candidate=*rit;
                    const uint32_t distance=target-candidate;
                    if (!distance || distance>window_pixels) continue;
                    const uint32_t src=candidate*channels;
                    uint32_t length=0;
                    const uint32_t limit=pixels-i;
                    while (length<limit &&
                           std::memcmp(combined.data()+src+uint64_t(length)*channels,
                                       combined.data()+local+uint64_t(length)*channels,channels)==0) {
                        ++length;
                    }
                    if (length>best_len) {
                        best_len=length; best_dist=distance;
                        if (length>=128) break;
                    }
                }
            }
        }
        uint32_t run=1;
        while (i+run<pixels && run<0x7fffffffU) {
            const uint32_t a=local+(run-1)*channels;
            const uint32_t b=local+run*channels;
            if (std::memcmp(combined.data()+a,combined.data()+b,channels)!=0) break;
            ++run;
        }
        const uint32_t match_cost=1+varint_size(best_dist)+varint_size(best_len);
        const uint32_t run_cost=1+channels+varint_size(run);
        const uint64_t literal_cost=uint64_t(best_len)*channels;
        const bool use_match=best_len>=4 && match_cost<literal_cost && best_len>=run;
        const bool use_run=!use_match && run>=3 && run_cost<uint64_t(run)*channels;
        if (use_match || use_run) {
            flush_literals();
            if (use_match) {
                tokens.push_back(2); write_varint(tokens,best_dist); write_varint(tokens,best_len);
                i += best_len;
            } else {
                tokens.push_back(1); write_varint(tokens,run);
                tokens.insert(tokens.end(),combined.begin()+local,combined.begin()+local+channels);
                i += run;
            }
        } else {
            if (!literal_count) literal_start=i;
            ++literal_count;
            ++i;
        }
        if (tokens.size()>cap) return -2;
    }
    flush_literals();
    if (tokens.size()>cap) return -2;
    std::memcpy(out,tokens.data(),tokens.size());
    *out_n=uint32_t(tokens.size());
    return 0;
}

int pixel_lz_decode_impl(const uint8_t* tokens, uint32_t tokens_n,
                         const uint8_t* history, uint32_t history_n,
                         uint32_t expected_pixels, uint32_t channels,
                         uint8_t* out, uint32_t out_n) {
    if (!tokens || !out || channels<1 || channels>4 || history_n%channels ||
        uint64_t(expected_pixels)*channels!=out_n) return -1;
    uint32_t pos=0, optr=0;
    while (pos<tokens_n) {
        const uint8_t op=tokens[pos++];
        if (op==0) {
            uint64_t count=0;
            if (!read_varint(tokens,tokens_n,pos,count) || count<1 || count>(out_n-optr)/channels) return -2;
            const uint64_t n=count*channels;
            if (pos+n>tokens_n || optr+n>out_n) return -3;
            std::memcpy(out+optr,tokens+pos,size_t(n)); pos+=uint32_t(n); optr+=uint32_t(n);
        } else if (op==1) {
            uint64_t count=0;
            if (!read_varint(tokens,tokens_n,pos,count) || count<3 || pos+channels>tokens_n || count>(out_n-optr)/channels) return -4;
            for (uint64_t i=0;i<count;++i) std::memcpy(out+optr+i*channels,tokens+pos,channels);
            pos+=channels; optr+=uint32_t(count*channels);
        } else if (op==2) {
            uint64_t distance=0,count=0;
            if (!read_varint(tokens,tokens_n,pos,distance) || !read_varint(tokens,tokens_n,pos,count) ||
                distance<1 || count<4 || count>(out_n-optr)/channels) return -5;
            const uint64_t available=uint64_t(history_n)/channels + optr/channels;
            if (distance>available) return -6;
            for (uint64_t i=0;i<count;++i) {
                const int64_t source=int64_t(history_n)+int64_t(optr)-int64_t(distance*channels);
                if (source<0) {
                    const int64_t hs=int64_t(history_n)+int64_t(optr)-int64_t(distance*channels);
                    if (hs<0 || uint64_t(hs)+channels>history_n) return -7;
                    std::memcpy(out+optr,history+hs,channels);
                } else {
                    if (uint64_t(source)+channels>uint64_t(history_n)+optr) return -8;
                    if (uint64_t(source)<history_n) std::memcpy(out+optr,history+source,channels);
                    else std::memcpy(out+optr,out+(source-int64_t(history_n)),channels);
                }
                optr+=channels;
            }
        } else return -9;
    }
    return optr==out_n ? 0 : -10;
}


class PixelLZNativeState {
public:
    PixelLZNativeState(uint32_t channels, uint32_t window_pixels)
        : channels_(channels), window_(std::max<uint32_t>(3, window_pixels)) {
        if (channels_ < 1 || channels_ > 4) throw std::invalid_argument("invalid channel count");
    }

    void seed(const uint8_t* history, uint32_t history_n, uint64_t total_pixels) {
        if (history_n % channels_) throw std::invalid_argument("unaligned history");
        if (history_n) history_.assign(history, history + history_n);
        else history_.clear();
        const size_t limit = size_t(window_) * channels_;
        if (history_.size() > limit) history_.erase(history_.begin(), history_.end() - limit);
        total_pixels_ = total_pixels;
        indexed_until_ = total_pixels_ - history_.size() / channels_;
        since_rebuild_ = 0;
        table_ready_ = false;
        keys_.clear(); heads_.clear(); links_.clear(); hashes_.clear(); positions_.clear();
    }

    void update(const uint8_t* raw, uint32_t raw_n) {
        if (raw_n % channels_) throw std::invalid_argument("unaligned raw pixels");
        if (raw_n) history_.insert(history_.end(), raw, raw + raw_n);
        const uint64_t pixels = raw_n / channels_;
        total_pixels_ += pixels;
        const size_t limit = size_t(window_) * channels_;
        if (history_.size() > limit) history_.erase(history_.begin(), history_.end() - limit);
    }

    int encode(const uint8_t* raw, uint32_t raw_n, uint8_t* out, uint32_t cap, uint32_t* out_n) {
        if (!raw || !out || !out_n || raw_n % channels_) return -1;
        ensure_table();
        const uint32_t hist_n = uint32_t(history_.size());
        const uint64_t hist_pixels = hist_n / channels_;
        const uint64_t pixels = raw_n / channels_;
        const uint64_t base_abs = total_pixels_ - hist_pixels;
        const uint64_t end_abs = total_pixels_ + pixels;
        std::vector<uint8_t> combined(size_t(hist_n) + raw_n);
        if (hist_n) std::memcpy(combined.data(), history_.data(), hist_n);
        if (raw_n) std::memcpy(combined.data() + hist_n, raw, raw_n);

        if (since_rebuild_ > window_ / 2) {
            rebuild(std::max<uint64_t>(base_abs, total_pixels_ > window_ ? total_pixels_ - window_ : 0), indexed_until_);
            since_rebuild_ = 0;
        }

        std::vector<uint8_t> tokens;
        tokens.reserve(size_t(raw_n) + size_t(pixels) / 8 + 64);
        uint64_t literal_start = 0;
        uint64_t literal_count = 0;
        auto flush_literals = [&]() {
            if (!literal_count) return;
            tokens.push_back(0);
            write_varint(tokens, literal_count);
            const size_t start = size_t(hist_pixels + literal_start) * channels_;
            const size_t count = size_t(literal_count) * channels_;
            tokens.insert(tokens.end(), combined.begin() + start, combined.begin() + start + count);
            literal_count = 0;
        };

        uint64_t i = 0;
        while (i < pixels) {
            const uint64_t target_abs = total_pixels_ + i;
            index_range(combined, base_abs, target_abs);
            const size_t local = size_t(hist_pixels + i) * channels_;
            uint64_t best_len = 0, best_dist = 0;
            if (i + 3 <= pixels) {
                const uint64_t key = pixel_hash(combined.data() + local, channels_);
                const uint32_t slot = find_slot(key);
                int64_t candidate = heads_[slot];
                int checked = 0;
                const uint64_t max_dist = std::min<uint64_t>(window_, target_abs - base_abs);
                while (candidate >= 0 && checked < 10) {
                    const uint64_t cand = uint64_t(candidate);
                    const size_t ring = size_t(cand % window_);
                    if (positions_[ring] != candidate) break;
                    const uint64_t distance = target_abs - cand;
                    if (distance > 0 && distance <= max_dist && hashes_[ring] == key) {
                        const size_t src = size_t(cand - base_abs) * channels_;
                        uint64_t length = 0;
                        const uint64_t limit = pixels - i;
                        bool match = true;
                        while (length < limit && match) {
                            for (uint32_t c = 0; c < channels_; ++c) {
                                if (combined[src + size_t(length) * channels_ + c] !=
                                    combined[local + size_t(length) * channels_ + c]) {
                                    match = false; break;
                                }
                            }
                            if (match) ++length;
                        }
                        if (length > best_len) {
                            best_len = length; best_dist = distance;
                            if (length >= 128) break;
                        }
                    }
                    candidate = links_[ring];
                    ++checked;
                }
            }

            uint64_t run = 1;
            while (i + run < pixels) {
                const size_t a = local + size_t(run - 1) * channels_;
                const size_t b = local + size_t(run) * channels_;
                if (std::memcmp(combined.data() + a, combined.data() + b, channels_) != 0) break;
                ++run;
            }
            const uint64_t match_cost = 1 + varint_size(best_dist) + varint_size(best_len);
            const uint64_t run_cost = 1 + channels_ + varint_size(run);
            const uint64_t literal_cost = best_len * channels_;
            const bool use_match = best_len >= 4 && match_cost < literal_cost && best_len >= run;
            const bool use_run = !use_match && run >= 3 && run_cost < run * channels_;
            if (use_match || use_run) {
                flush_literals();
                if (use_match) {
                    tokens.push_back(2); write_varint(tokens, best_dist); write_varint(tokens, best_len);
                    i += best_len;
                } else {
                    tokens.push_back(1); write_varint(tokens, run);
                    tokens.insert(tokens.end(), combined.begin() + local, combined.begin() + local + channels_);
                    i += run;
                }
            } else {
                if (!literal_count) literal_start = i;
                ++literal_count; ++i;
            }
            if (tokens.size() > cap) return -2;
        }
        flush_literals();
        if (tokens.size() > cap) return -2;
        // Keep the two final pixels pending: their three-pixel hash reaches into
        // the next tile and is inserted once those following pixels exist.
        index_range(combined, base_abs, end_abs);
        std::memcpy(out, tokens.data(), tokens.size());
        *out_n = uint32_t(tokens.size());
        total_pixels_ = end_abs;
        history_.assign(combined.end() - std::min<size_t>(combined.size(), size_t(window_) * channels_), combined.end());
        return 0;
    }

private:
    uint32_t channels_, window_;
    uint32_t mask_ = 0;
    uint64_t total_pixels_ = 0, indexed_until_ = 0;
    uint64_t since_rebuild_ = 0;
    bool table_ready_ = false;
    std::vector<uint8_t> history_;
    std::vector<uint64_t> keys_, hashes_;
    std::vector<int64_t> heads_, links_, positions_;

    void ensure_table() {
        if (table_ready_) return;
        uint64_t target = std::max<uint64_t>(uint64_t(window_) * 2, 1u << 18);
        uint64_t size = 1;
        while (size < target) size <<= 1;
        if (size > (1u << 25)) throw std::bad_alloc();
        mask_ = uint32_t(size - 1);
        keys_.assign(size, 0);
        heads_.assign(size, -1);
        links_.assign(window_, -1);
        hashes_.assign(window_, 0);
        positions_.assign(window_, -1);
        table_ready_ = true;
    }

    uint32_t find_slot(uint64_t key) const {
        uint32_t slot = uint32_t(key & mask_);
        while (keys_[slot] != 0 && keys_[slot] != key) {
            slot = (slot + 1) & mask_;
        }
        return slot;
    }

    void insert(const std::vector<uint8_t>& combined, uint64_t base_abs, uint64_t absolute) {
        const uint64_t relative = absolute - base_abs;
        const size_t offset = size_t(relative) * channels_;
        if (offset + size_t(3) * channels_ > combined.size()) return;
        const uint64_t key = pixel_hash(combined.data() + offset, channels_);
        const uint32_t slot = find_slot(key);
        const size_t ring = size_t(absolute % window_);
        links_[ring] = heads_[slot];
        hashes_[ring] = key;
        positions_[ring] = int64_t(absolute);
        if (keys_[slot] == 0) keys_[slot] = key;
        heads_[slot] = int64_t(absolute);
    }

    void index_range(const std::vector<uint8_t>& combined, uint64_t base_abs, uint64_t end_abs) {
        const uint64_t total_in_combined = base_abs + combined.size() / channels_;
        const uint64_t low = std::max<uint64_t>(base_abs, total_pixels_ > window_ ? total_pixels_ - window_ : 0);
        uint64_t start = std::max(indexed_until_, low);
        const uint64_t end = std::min<uint64_t>(end_abs, total_in_combined >= 2 ? total_in_combined - 2 : 0);
        if (start > end) start = end;
        for (uint64_t absolute = start; absolute < end; ++absolute) {
            insert(combined, base_abs, absolute);
            ++since_rebuild_;
        }
        indexed_until_ = std::max(indexed_until_, end);
    }

    void rebuild(uint64_t low_abs, uint64_t high_abs) {
        std::fill(keys_.begin(), keys_.end(), 0);
        std::fill(heads_.begin(), heads_.end(), -1);
        for (uint64_t absolute = low_abs; absolute < high_abs; ++absolute) {
            const size_t ring = size_t(absolute % window_);
            if (positions_[ring] != int64_t(absolute) || hashes_[ring] == 0) continue;
            const uint64_t key = hashes_[ring];
            const uint32_t slot = find_slot(key);
            links_[ring] = heads_[slot];
            if (keys_[slot] == 0) keys_[slot] = key;
            heads_[slot] = int64_t(absolute);
        }
    }
 };

class BlockNativeState {
public:
    explicit BlockNativeState(uint32_t channels, uint32_t limit=32768)
        : channels_(channels), limit_(limit) {
        if (channels_<1 || channels_>4 || !limit_) throw std::invalid_argument("invalid block state parameters");
    }

    int encode(const uint8_t* raw, uint32_t width, uint32_t height,
               uint8_t* out, uint32_t cap, uint32_t* out_n) {
        if (!raw || !out || !out_n || !width || !height) return -1;
        std::vector<uint8_t> tokens;
        const uint32_t stride=width*channels_;
        for (uint32_t by=0;by<height;by+=8) {
            const uint32_t bh=std::min<uint32_t>(8,height-by);
            for (uint32_t bx=0;bx<width;bx+=8) {
                const uint32_t bw=std::min<uint32_t>(8,width-bx);
                std::vector<uint8_t> block(size_t(bw)*bh*channels_);
                size_t p=0;
                for (uint32_t y=0;y<bh;++y) {
                    const uint8_t* src=raw+uint64_t(by+y)*stride+uint64_t(bx)*channels_;
                    const size_t n=size_t(bw)*channels_;
                    std::memcpy(block.data()+p,src,n); p+=n;
                }
                const uint64_t key=block_hash(bw,bh,block);
                uint64_t best=UINT64_MAX;
                auto it=heads_.find(key);
                if (it!=heads_.end()) {
                    const auto& chain=it->second;
                    for (auto rit=chain.rbegin();rit!=chain.rend();++rit) {
                        const BlockItem* item=get(*rit);
                        if (item && item->w==bw && item->h==bh && item->bytes==block) {
                            best=*rit; break;
                        }
                    }
                }
                if (best!=UINT64_MAX) {
                    tokens.push_back(1); write_varint(tokens,next_id_-best);
                } else {
                    tokens.push_back(0); tokens.insert(tokens.end(),block.begin(),block.end());
                }
                add(bw,bh,std::move(block));
                if (tokens.size()>cap) return -2;
            }
        }
        if (tokens.size()>cap) return -2;
        std::memcpy(out,tokens.data(),tokens.size()); *out_n=uint32_t(tokens.size());
        return 0;
    }

    int update(const uint8_t* raw, uint32_t width, uint32_t height) {
        if (!raw || !width || !height) return -1;
        const uint32_t stride=width*channels_;
        for (uint32_t by=0;by<height;by+=8) {
            const uint32_t bh=std::min<uint32_t>(8,height-by);
            for (uint32_t bx=0;bx<width;bx+=8) {
                const uint32_t bw=std::min<uint32_t>(8,width-bx);
                std::vector<uint8_t> block(size_t(bw)*bh*channels_);
                size_t p=0;
                for (uint32_t y=0;y<bh;++y) {
                    const uint8_t* src=raw+uint64_t(by+y)*stride+uint64_t(bx)*channels_;
                    const size_t n=size_t(bw)*channels_;
                    std::memcpy(block.data()+p,src,n); p+=n;
                }
                add(bw,bh,std::move(block));
            }
        }
        return 0;
    }

    int decode(const uint8_t* tokens, uint32_t tokens_n, uint32_t width, uint32_t height,
               uint8_t* out, uint32_t out_n) {
        if (!tokens || !out || !width || !height || uint64_t(width)*height*channels_!=out_n) return -1;
        uint32_t pos=0;
        const uint32_t stride=width*channels_;
        for (uint32_t by=0;by<height;by+=8) {
            const uint32_t bh=std::min<uint32_t>(8,height-by);
            for (uint32_t bx=0;bx<width;bx+=8) {
                const uint32_t bw=std::min<uint32_t>(8,width-bx);
                if (pos>=tokens_n) return -2;
                const uint8_t op=tokens[pos++];
                std::vector<uint8_t> block;
                const size_t block_n=size_t(bw)*bh*channels_;
                if (op==0) {
                    if (pos+block_n>tokens_n) return -3;
                    block.assign(tokens+pos,tokens+pos+block_n); pos+=uint32_t(block_n);
                } else if (op==1) {
                    uint64_t distance=0; uint32_t shift=0; bool done=false;
                    for (int k=0;k<10;++k) {
                        if (pos>=tokens_n || shift>=64) return -4;
                        const uint8_t b=tokens[pos++]; distance |= uint64_t(b&127)<<shift;
                        if (!(b&128)) { done=true; break; }
                        shift+=7;
                    }
                    if (!done || !distance || distance>next_id_) return -5;
                    const BlockItem* item=get(next_id_-distance);
                    if (!item || item->w!=bw || item->h!=bh || item->bytes.size()!=block_n) return -6;
                    block=item->bytes;
                } else return -7;
                size_t p=0;
                for (uint32_t y=0;y<bh;++y) {
                    uint8_t* dst=out+uint64_t(by+y)*stride+uint64_t(bx)*channels_;
                    const size_t n=size_t(bw)*channels_;
                    std::memcpy(dst,block.data()+p,n); p+=n;
                }
                add(bw,bh,std::move(block));
            }
        }
        return pos==tokens_n ? 0 : -8;
    }

private:
    struct BlockItem { uint32_t w,h; std::vector<uint8_t> bytes; };
    uint32_t channels_,limit_;
    uint64_t next_id_=0,base_id_=0;
    std::deque<BlockItem> items_;
    std::unordered_map<uint64_t,std::vector<uint64_t>> heads_;

    uint64_t block_hash(uint32_t w,uint32_t h,const std::vector<uint8_t>& bytes) const {
        uint64_t x=1469598103934665603ULL;
        auto mix=[&](uint8_t b){x^=b;x*=1099511628211ULL;};
        mix(uint8_t(w)); mix(uint8_t(h));
        for (uint8_t b:bytes) mix(b);
        return x ? x : 1;
    }
    const BlockItem* get(uint64_t id) const {
        if (id<base_id_ || id>=next_id_) return nullptr;
        const uint64_t offset=id-base_id_;
        return offset<items_.size() ? &items_[size_t(offset)] : nullptr;
    }
    void add(uint32_t w,uint32_t h,std::vector<uint8_t>&& bytes) {
        const uint64_t id=next_id_++;
        const uint64_t key=block_hash(w,h,bytes);
        items_.push_back(BlockItem{w,h,std::move(bytes)});
        auto& chain=heads_[key]; chain.push_back(id);
        if (chain.size()>12) chain.erase(chain.begin());
        if (items_.size()>limit_) {
            items_.pop_front(); ++base_id_;
            if ((id % 4096)==0) {
                for (auto it=heads_.begin();it!=heads_.end();) {
                    auto& q=it->second;
                    while (!q.empty() && q.front()<base_id_) q.erase(q.begin());
                    if (q.empty()) it=heads_.erase(it); else ++it;
                }
            }
        }
    }
};

int rans_decode_spatial_impl(const uint8_t* blob, uint32_t blob_n, uint32_t output_n,
                             uint32_t width, uint32_t height, uint32_t channels,
                             uint8_t* out) {
    if (!blob || !out || channels<1 || channels>4) return -1;
    const uint64_t rowbytes=uint64_t(width)*channels+1;
    if (!width || !height || rowbytes*height!=output_n) return -2;
    const uint32_t contexts=channels*4+1;
    if (blob_n<5 || blob[0]!=contexts) return -3;
    std::vector<uint16_t> freq(size_t(contexts)*256,0), cumul(size_t(contexts)*256,0);
    std::vector<uint8_t> decode(size_t(contexts)*RANS_TOTAL,0);
    uint32_t pos=1;
    for (uint32_t ctx=0;ctx<contexts;++ctx) {
        if (pos+2>blob_n) return -4;
        const uint32_t table_n=uint32_t(blob[pos]) | (uint32_t(blob[pos+1])<<8);
        pos+=2;
        if (!table_n) continue;
        if (pos+table_n>blob_n) return -5;
        const uint32_t end=pos+table_n;
        const uint8_t mode=blob[pos++];
        uint16_t* f=freq.data()+size_t(ctx)*256;
        if (mode==0) {
            if (pos+512!=end) return -6;
            for (uint32_t sym=0;sym<256;++sym) {
                f[sym]=uint16_t(blob[pos]) | (uint16_t(blob[pos+1])<<8);
                pos+=2;
            }
        } else if (mode==1) {
            if (pos>=end) return -7;
            uint32_t count=blob[pos++]; if (!count) count=256;
            int previous=-1;
            for (uint32_t k=0;k<count;++k) {
                if (pos>=end) return -8;
                const uint32_t sym=blob[pos++];
                if (int(sym)<=previous) return -9;
                previous=int(sym);
                uint64_t value=0; uint32_t shift=0; bool done=false;
                for (int j=0;j<5;++j) {
                    if (pos>=end || shift>=32) return -10;
                    const uint8_t b=blob[pos++];
                    value |= uint64_t(b&127)<<shift;
                    if (!(b&128)) { done=true; break; }
                    shift+=7;
                }
                if (!done || value<1 || value>RANS_TOTAL) return -11;
                f[sym]=uint16_t(value);
            }
            if (pos!=end) return -12;
        } else return -13;
        uint32_t sum=0;
        uint16_t* c=cumul.data()+size_t(ctx)*256;
        uint8_t* d=decode.data()+size_t(ctx)*RANS_TOTAL;
        for (uint32_t sym=0;sym<256;++sym) {
            c[sym]=uint16_t(sum);
            const uint32_t count=f[sym];
            if (sum+count>RANS_TOTAL) return -14;
            std::fill(d+sum,d+sum+count,uint8_t(sym));
            sum+=count;
        }
        if (sum!=RANS_TOTAL) return -15;
        pos=end;
    }
    uint32_t end_pos=0;
    const int rc=rans_decode(blob,blob_n,pos,output_n,5,width,height,channels,contexts,
                             freq.data(),cumul.data(),decode.data(),out,&end_pos);
    if (rc) return -20+rc;
    if (end_pos!=blob_n) return -40;
    return 0;
}

int filter_legacy_impl(const uint8_t* raw, uint32_t width, uint32_t height,
                       uint32_t channels, uint32_t transform,
                       uint8_t* out, uint32_t out_n) {
    if (!raw || !out || channels < 1 || channels > 4 || transform > 4) return -1;
    const uint64_t stride64 = uint64_t(width) * channels;
    const uint64_t expected = uint64_t(height) * (stride64 + 1);
    if (!width || !height || expected != out_n || stride64 > 0xffffffffULL) return -2;
    const uint32_t stride = uint32_t(stride64);
    const uint32_t src_size = stride * height;
    std::vector<uint8_t> transformed(src_size);
    for (uint32_t i = 0; i < width * height; ++i)
        forward_pixel(raw + uint64_t(i) * channels, transformed.data() + uint64_t(i) * channels,
                      int(channels), int(transform));
    std::vector<uint8_t> candidate(stride), best(stride);
    const double base_score = stride > 1 ? double(stride) * std::log2(double(stride)) : 0.0;
    for (uint32_t y = 0; y < height; ++y) {
        double best_score = std::numeric_limits<double>::infinity();
        uint8_t best_predictor = 0;
        const uint8_t* row = transformed.data() + uint64_t(y) * stride;
        const uint8_t* prev = y ? row - stride : nullptr;
        for (int pred = 0; pred < 7; ++pred) {
            std::array<uint32_t,256> hist{};
            for (uint32_t i = 0; i < stride; ++i) {
                const uint8_t left = i >= channels ? row[i-channels] : 0;
                const uint8_t up = prev ? prev[i] : 0;
                const uint8_t ul = (prev && i >= channels) ? prev[i-channels] : 0;
                const uint8_t residual = uint8_t(int(row[i]) - predictor(pred,left,up,ul));
                candidate[i] = residual;
                ++hist[residual];
            }
            double score = base_score;
            for (uint32_t sym = 0; sym < 256; ++sym) {
                const uint32_t count = hist[sym];
                if (count) score -= double(count) * std::log2(double(count));
            }
            if (score < best_score) {
                best_score = score; best_predictor = uint8_t(pred); best = candidate;
            }
        }
        const uint64_t dst = uint64_t(y) * (stride + 1);
        out[dst] = best_predictor;
        std::memcpy(out + dst + 1, best.data(), stride);
    }
    return 0;
}

int unfilter_legacy_impl(const uint8_t* filtered, uint32_t width, uint32_t height,
                         uint32_t channels, uint32_t transform,
                         uint8_t* out, uint32_t filtered_n, uint32_t out_n) {
    if (!filtered || !out || channels < 1 || channels > 4 || transform > 4) return -1;
    const uint64_t stride64 = uint64_t(width) * channels;
    if (!width || !height || stride64 > 0xffffffffULL ||
        uint64_t(height) * (stride64 + 1) != filtered_n || stride64 * height != out_n) return -2;
    const uint32_t stride = uint32_t(stride64), rowbytes = stride + 1;
    std::vector<uint8_t> transformed(out_n);
    for (uint32_t y = 0; y < height; ++y) {
        const uint64_t src = uint64_t(y) * rowbytes;
        uint8_t pred = filtered[src];
        if (pred > 6) return -3;
        uint8_t* row = transformed.data() + uint64_t(y) * stride;
        const uint8_t* prev = y ? row - stride : nullptr;
        for (uint32_t i = 0; i < stride; ++i) {
            const uint8_t left = i >= channels ? row[i-channels] : 0;
            const uint8_t up = prev ? prev[i] : 0;
            const uint8_t ul = (prev && i >= channels) ? prev[i-channels] : 0;
            row[i] = uint8_t(int(filtered[src + 1 + i]) + predictor(pred,left,up,ul));
        }
    }
    for (uint32_t i = 0; i < width * height; ++i)
        inverse_pixel(transformed.data() + uint64_t(i) * channels, out + uint64_t(i) * channels,
                      int(channels), int(transform));
    return 0;
}

void normalize_counts(const std::array<uint64_t,256>& counts, uint16_t* freq) {
    std::fill(freq, freq + 256, uint16_t(0));
    uint64_t total = 0;
    std::vector<uint32_t> present;
    std::vector<std::pair<uint64_t,uint32_t>> remainders;
    for (uint32_t s=0;s<256;++s) {
        total += counts[s];
        if (counts[s]) present.push_back(s);
    }
    if (!total || present.empty()) throw std::runtime_error("empty rANS context");
    remainders.reserve(present.size());
    uint32_t sum=0;
    for (uint32_t sym:present) {
        const uint64_t scaled=counts[sym]*RANS_TOTAL;
        uint32_t q=uint32_t(scaled/total);
        uint64_t r=scaled%total;
        if (q<1) q=1;
        freq[sym]=uint16_t(q);
        sum += q;
        remainders.emplace_back(r,sym);
    }
    int64_t delta=int64_t(RANS_TOTAL)-int64_t(sum);
    if (delta>0) {
        std::sort(remainders.begin(),remainders.end(),[](const auto& a,const auto& b){
            return a.first>b.first || (a.first==b.first && a.second>b.second);
        });
        for (int64_t i=0;i<delta;++i) ++freq[remainders[size_t(i)%remainders.size()].second];
    } else if (delta<0) {
        std::array<uint64_t,256> remainder_by_symbol{};
        for (auto x:remainders) remainder_by_symbol[x.second]=x.first;
        uint64_t need=uint64_t(-delta);
        while (need) {
            std::vector<uint32_t> candidates;
            for (uint32_t sym:present) if (freq[sym]>1) candidates.push_back(sym);
            if (candidates.empty()) throw std::runtime_error("frequency normalization underflow");
            std::sort(candidates.begin(),candidates.end(),[&](uint32_t a,uint32_t b){
                if (remainder_by_symbol[a] != remainder_by_symbol[b])
                    return remainder_by_symbol[a] < remainder_by_symbol[b];
                if (freq[a] != freq[b]) return freq[a] > freq[b];
                return a < b;
            });
            for (uint32_t sym:candidates) {
                if (!need) break;
                --freq[sym]; --need;
            }
        }
    }
}

void append_le16(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back(uint8_t(v)); out.push_back(uint8_t(v >> 8));
}
void append_le32(std::vector<uint8_t>& out, uint32_t v) {
    for (int i=0;i<4;++i) out.push_back(uint8_t(v >> (8*i)));
}

int rans_encode_spatial_impl(const uint8_t* data, uint32_t n, uint32_t width,
                             uint32_t height, uint32_t channels,
                             uint8_t* out, uint32_t cap, uint32_t* out_n) {
    if (!data || !out || !out_n || channels<1 || channels>4) return -1;
    const uint64_t rowbytes64=uint64_t(width)*channels+1;
    if (!width || !height || rowbytes64*height!=n) return -2;
    const uint32_t rowbytes=uint32_t(rowbytes64), contexts=channels*4+1;
    std::vector<uint8_t> ids(n);
    std::vector<std::array<uint64_t,256>> hist(contexts);
    for (auto& h:hist) h.fill(0);
    for (uint32_t i=0;i<n;++i) {
        const uint32_t col=i%rowbytes;
        uint32_t ctx=channels*4;
        if (col) {
            const uint32_t sample_pos=col-1;
            const uint32_t ch=sample_pos%channels;
            const uint8_t previous=sample_pos>=channels ? data[i-channels] : 0;
            ctx=ch*4+(previous>>6);
        }
        ids[i]=uint8_t(ctx);
        ++hist[ctx][data[i]];
    }
    std::vector<uint16_t> freq(size_t(contexts)*256), cumul(size_t(contexts)*256);
    std::vector<uint8_t> table;
    table.push_back(uint8_t(contexts));
    for (uint32_t ctx=0;ctx<contexts;++ctx) {
        const uint64_t total=std::accumulate(hist[ctx].begin(),hist[ctx].end(),uint64_t(0));
        if (!total) { append_le16(table,0); continue; }
        uint16_t* f=freq.data()+size_t(ctx)*256;
        uint16_t* c=cumul.data()+size_t(ctx)*256;
        normalize_counts(hist[ctx],f);
        uint32_t sum=0;
        for (uint32_t sym=0;sym<256;++sym) { c[sym]=uint16_t(sum); sum+=f[sym]; }
        std::vector<uint8_t> dense, sparse;
        dense.reserve(513); dense.push_back(0);
        for (uint32_t sym=0;sym<256;++sym) append_le16(dense,f[sym]);
        std::vector<uint32_t> present;
        for (uint32_t sym=0;sym<256;++sym) if (f[sym]) present.push_back(sym);
        sparse.push_back(1); sparse.push_back(uint8_t(present.size() & 255));
        for (uint32_t sym:present) {
            sparse.push_back(uint8_t(sym));
            uint32_t v=f[sym];
            while (v>=128) { sparse.push_back(uint8_t((v&127)|128)); v >>= 7; }
            sparse.push_back(uint8_t(v));
        }
        const auto& chosen = sparse.size()<dense.size() ? sparse : dense;
        if (chosen.size()>65535) return -3;
        append_le16(table,uint16_t(chosen.size()));
        table.insert(table.end(),chosen.begin(),chosen.end());
    }
    std::vector<uint8_t> emitted(std::max<uint64_t>(64,uint64_t(n)*2+32));
    uint32_t emitted_n=0,state=0;
    const int rc=rans_encode_ids(data,ids.data(),n,freq.data(),cumul.data(),
                                 emitted.data(),uint32_t(emitted.size()),&emitted_n,&state);
    if (rc) return -4;
    const uint64_t final_size=table.size()+4+emitted_n;
    if (final_size>cap) return -5;
    std::memcpy(out,table.data(),table.size());
    uint32_t pos=uint32_t(table.size());
    out[pos++]=uint8_t(state); out[pos++]=uint8_t(state>>8);
    out[pos++]=uint8_t(state>>16); out[pos++]=uint8_t(state>>24);
    if (emitted_n) std::memcpy(out+pos,emitted.data(),emitted_n);
    *out_n=uint32_t(final_size);
    return 0;
}

} // namespace

extern "C" {
PXL_EXPORT int pxl_native_abi_version() { return 5; }

PXL_EXPORT void* pxl_block_state_create(uint32_t channels, uint32_t history_limit) {
    try { return new BlockNativeState(channels,history_limit); } catch (...) { return nullptr; }
}
PXL_EXPORT void* pxl_block_state_clone(void* state) {
    if (!state) return nullptr;
    try { return new BlockNativeState(*static_cast<BlockNativeState*>(state)); } catch (...) { return nullptr; }
}
PXL_EXPORT void pxl_block_state_destroy(void* state) { delete static_cast<BlockNativeState*>(state); }
PXL_EXPORT int pxl_block_state_encode(void* state,const uint8_t* raw,uint32_t width,uint32_t height,
                                      uint8_t* out,uint32_t cap,uint32_t* out_n) {
    if (!state) return -1;
    try { return static_cast<BlockNativeState*>(state)->encode(raw,width,height,out,cap,out_n); }
    catch (...) { return -10; }
}
PXL_EXPORT int pxl_block_state_update(void* state,const uint8_t* raw,uint32_t width,uint32_t height) {
    if (!state) return -1;
    try { return static_cast<BlockNativeState*>(state)->update(raw,width,height); }
    catch (...) { return -10; }
}
PXL_EXPORT int pxl_block_state_decode(void* state,const uint8_t* tokens,uint32_t tokens_n,
                                      uint32_t width,uint32_t height,uint8_t* out,uint32_t out_n) {
    if (!state) return -1;
    try { return static_cast<BlockNativeState*>(state)->decode(tokens,tokens_n,width,height,out,out_n); }
    catch (...) { return -10; }
}

PXL_EXPORT void* pxl_pixel_lz_state_create(uint32_t channels, uint32_t window_pixels) {
    try { return new PixelLZNativeState(channels, window_pixels); }
    catch (...) { return nullptr; }
}
PXL_EXPORT void pxl_pixel_lz_state_destroy(void* state) {
    delete static_cast<PixelLZNativeState*>(state);
}
PXL_EXPORT int pxl_pixel_lz_state_seed(void* state, const uint8_t* history, uint32_t history_n,
                                      uint64_t total_pixels) {
    if (!state) return -1;
    try { static_cast<PixelLZNativeState*>(state)->seed(history, history_n, total_pixels); return 0; }
    catch (...) { return -2; }
}
PXL_EXPORT int pxl_pixel_lz_state_update(void* state, const uint8_t* raw, uint32_t raw_n) {
    if (!state) return -1;
    try { static_cast<PixelLZNativeState*>(state)->update(raw, raw_n); return 0; }
    catch (...) { return -2; }
}
PXL_EXPORT int pxl_pixel_lz_state_encode(void* state, const uint8_t* raw, uint32_t raw_n,
                                        uint8_t* out, uint32_t cap, uint32_t* out_n) {
    if (!state) return -1;
    try { return static_cast<PixelLZNativeState*>(state)->encode(raw, raw_n, out, cap, out_n); }
    catch (...) { return -3; }
}
PXL_EXPORT int pxl_filter_legacy(const uint8_t* raw, uint32_t width, uint32_t height,
                                 uint32_t channels, uint32_t transform,
                                 uint8_t* out, uint32_t out_n) {
    try { return filter_legacy_impl(raw,width,height,channels,transform,out,out_n); }
    catch (...) { return -4; }
}
PXL_EXPORT int pxl_unfilter_legacy(const uint8_t* filtered, uint32_t width, uint32_t height,
                                   uint32_t channels, uint32_t transform,
                                   uint8_t* out, uint32_t filtered_n, uint32_t out_n) {
    try { return unfilter_legacy_impl(filtered,width,height,channels,transform,out,filtered_n,out_n); }
    catch (...) { return -4; }
}
PXL_EXPORT int pxl_rans_encode_spatial(const uint8_t* data, uint32_t n, uint32_t width,
                                       uint32_t height, uint32_t channels,
                                       uint8_t* out, uint32_t cap, uint32_t* out_n) {
    try { return rans_encode_spatial_impl(data,n,width,height,channels,out,cap,out_n); }
    catch (...) { return -6; }
}
PXL_EXPORT int pxl_rans_decode_spatial(const uint8_t* blob, uint32_t blob_n, uint32_t output_n,
                                       uint32_t width, uint32_t height, uint32_t channels,
                                       uint8_t* out) {
    try { return rans_decode_spatial_impl(blob,blob_n,output_n,width,height,channels,out); }
    catch (...) { return -50; }
}

PXL_EXPORT int pxl_pixel_lz_encode(const uint8_t* history, uint32_t history_n,
                                   const uint8_t* raw, uint32_t raw_n, uint32_t channels,
                                   uint32_t window_pixels, uint8_t* out, uint32_t cap,
                                   uint32_t* out_n) {
    return pixel_lz_encode_impl(history,history_n,raw,raw_n,channels,window_pixels,out,cap,out_n);
}
PXL_EXPORT int pxl_pixel_lz_decode(const uint8_t* tokens, uint32_t tokens_n,
                                   const uint8_t* history, uint32_t history_n,
                                   uint32_t expected_pixels, uint32_t channels,
                                   uint8_t* out, uint32_t out_n) {
    return pixel_lz_decode_impl(tokens,tokens_n,history,history_n,expected_pixels,channels,out,out_n);
}

PXL_EXPORT int pxl_rans_encode_ids(const uint8_t* data, const uint8_t* ids, uint32_t n,
                                   const uint16_t* freq, const uint16_t* cumul,
                                   uint8_t* out, uint32_t cap, uint32_t* out_n,
                                   uint32_t* out_state) {
    return rans_encode_ids(data, ids, n, freq, cumul, out, cap, out_n, out_state);
}

PXL_EXPORT int pxl_rans_encode_context16(const uint8_t* data, uint32_t n,
                                         const uint16_t* freq, const uint16_t* cumul,
                                         uint8_t* out, uint32_t cap, uint32_t* out_n,
                                         uint32_t* out_state) {
    if (!data) return -1;
    std::vector<uint8_t> ids(n);
    for (uint32_t i=0;i<n;++i) ids[i] = i ? uint8_t(data[i-1] >> 4) : 0;
    return rans_encode_ids(data, ids.data(), n, freq, cumul, out, cap, out_n, out_state);
}

PXL_EXPORT int pxl_rans_decode_context16(const uint8_t* blob, uint32_t blob_n, uint32_t start,
                                         uint32_t output_n, const uint16_t* freq,
                                         const uint16_t* cumul, const uint8_t* decode,
                                         uint8_t* out, uint32_t* end_pos) {
    return rans_decode(blob, blob_n, start, output_n, 16, 0, 0, 0, 16,
                       freq, cumul, decode, out, end_pos);
}

PXL_EXPORT int pxl_rans_decode_context2d(const uint8_t* blob, uint32_t blob_n, uint32_t start,
                                         uint32_t output_n, uint32_t width, uint32_t height,
                                         uint32_t channels, const uint16_t* freq,
                                         const uint16_t* cumul, const uint8_t* decode,
                                         uint8_t* out, uint32_t* end_pos) {
    if (height * (width * channels + channels) != output_n) return -8;
    return rans_decode(blob, blob_n, start, output_n, 2, width, height, channels, 17,
                       freq, cumul, decode, out, end_pos);
}

PXL_EXPORT int pxl_rans_decode_context2d_signed(const uint8_t* blob, uint32_t blob_n, uint32_t start,
                                                uint32_t output_n, uint32_t width, uint32_t height,
                                                uint32_t channels, const uint16_t* freq,
                                                const uint16_t* cumul, const uint8_t* decode,
                                                uint8_t* out, uint32_t* end_pos) {
    if (height * (width * channels + channels) != output_n) return -8;
    return rans_decode(blob, blob_n, start, output_n, 3, width, height, channels, 17,
                       freq, cumul, decode, out, end_pos);
}

PXL_EXPORT int pxl_rans_decode_context2d_triple(const uint8_t* blob, uint32_t blob_n, uint32_t start,
                                                uint32_t output_n, uint32_t width, uint32_t height,
                                                uint32_t channels, const uint16_t* freq,
                                                const uint16_t* cumul, const uint8_t* decode,
                                                uint8_t* out, uint32_t* end_pos) {
    if (height * (width * channels + channels) != output_n) return -8;
    return rans_decode(blob, blob_n, start, output_n, 4, width, height, channels, 65,
                       freq, cumul, decode, out, end_pos);
}

// Fused reversible color transform + v3 per-component, per-row predictor selection.
PXL_EXPORT int pxl_filter_color_v3(const uint8_t* raw, uint32_t width, uint32_t height,
                                  uint32_t channels, uint32_t transform, uint8_t* out,
                                  uint32_t out_n, uint8_t* ctx_ids) {
    if (!raw || !out || width == 0 || height == 0 || channels < 1 || channels > 4 || transform > 4) return -1;
    const uint64_t expected = uint64_t(height) * (uint64_t(width) * channels + channels);
    if (expected != out_n) return -2;
    const uint32_t stride = width * channels;
    std::vector<uint8_t> prev(stride, 0), curr(stride), residual(stride), prev_residual(stride, 0);
    uint32_t p = 0;
    for (uint32_t y=0; y<height; ++y) {
        const uint8_t* srcrow = raw + uint64_t(y) * stride;
        for (uint32_t x=0; x<width; ++x)
            forward_pixel(srcrow + x*channels, curr.data()+x*channels, int(channels), int(transform));
        for (uint32_t ch=0; ch<channels; ++ch) {
            std::array<uint32_t,256> hist{};
            std::array<uint64_t,7> magnitude{};
            std::vector<uint8_t> candidates[7];
            for (int pred=0; pred<7; ++pred) candidates[pred].resize(width);
            for (uint32_t x=0; x<width; ++x) {
                const uint8_t value = curr[x*channels+ch];
                const uint8_t left = x ? curr[(x-1)*channels+ch] : 0;
                const uint8_t up = prev[x*channels+ch];
                const uint8_t ul = x ? prev[(x-1)*channels+ch] : 0;
                for (int pred=0; pred<7; ++pred) {
                    const uint8_t r = uint8_t(int(value) - predictor(pred,left,up,ul));
                    candidates[pred][x] = r;
                }
            }
            double best_entropy = std::numeric_limits<double>::infinity();
            uint64_t best_mag = std::numeric_limits<uint64_t>::max();
            int best_id = 0;
            for (int pred=0; pred<7; ++pred) {
                hist.fill(0);
                uint64_t mag=0;
                for (uint32_t x=0; x<width; ++x) {
                    const uint8_t v=candidates[pred][x];
                    ++hist[v];
                    mag += std::min<uint32_t>(v, 256u-v);
                }
                double sum=0.0;
                for (uint32_t v : hist) if (v) sum += double(v)*std::log2(double(v));
                const double entropy = double(width)*std::log2(double(width)) - sum;
                if (entropy < best_entropy || (entropy == best_entropy && mag < best_mag)) {
                    best_entropy=entropy; best_mag=mag; best_id=pred;
                }
            }
            out[p+ch] = uint8_t(best_id);
            if (ctx_ids) ctx_ids[p+ch] = 16;
            for (uint32_t x=0; x<width; ++x) {
                const uint32_t ri = x*channels+ch;
                residual[ri] = candidates[best_id][x];
                if (ctx_ids) {
                    const uint8_t left_r = x ? residual[ri-channels] : 0;
                    const uint8_t up_r = prev_residual[ri];
                    ctx_ids[p+channels+ri] = uint8_t(((left_r >> 6) << 2) | (up_r >> 6));
                }
            }
        }
        p += channels;
        std::memcpy(out+p, residual.data(), stride);
        p += stride;
        prev.swap(curr);
        prev_residual.swap(residual);
    }
    return 0;
}

PXL_EXPORT int pxl_unfilter_color_v3(const uint8_t* filtered, uint32_t width, uint32_t height,
                                    uint32_t channels, uint32_t transform, uint8_t* out,
                                    uint32_t filtered_n, uint32_t out_n) {
    if (!filtered || !out || width == 0 || height == 0 || channels < 1 || channels > 4 || transform > 4) return -1;
    const uint64_t row_size = uint64_t(width)*channels + channels;
    if (uint64_t(height)*row_size != filtered_n || uint64_t(height)*width*channels != out_n) return -2;
    const uint32_t stride=width*channels;
    std::vector<uint8_t> prev(stride,0), curr(stride);
    uint32_t pos=0;
    for (uint32_t y=0;y<height;++y) {
        const uint8_t* pred_ids=filtered+pos; pos+=channels;
        const uint8_t* residual=filtered+pos; pos+=stride;
        for (uint32_t x=0;x<width;++x) {
            for (uint32_t ch=0;ch<channels;++ch) {
                const uint32_t i=x*channels+ch;
                if (pred_ids[ch] > 6) return -3;
                const uint8_t left=x ? curr[i-channels] : 0;
                const uint8_t up=prev[i];
                const uint8_t ul=x ? prev[i-channels] : 0;
                curr[i]=uint8_t(int(residual[i])+predictor(pred_ids[ch],left,up,ul));
            }
        }
        uint8_t* dst=out+uint64_t(y)*stride;
        for (uint32_t x=0;x<width;++x)
            inverse_pixel(curr.data()+x*channels,dst+x*channels,int(channels),int(transform));
        prev.swap(curr);
    }
    return 0;
}
}
