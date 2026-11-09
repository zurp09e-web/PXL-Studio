// PXL v8 native extensions: blended rANS, delta block references and SIMD dispatch.
// Kept in a separate translation unit so older codec primitives stay untouched.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
# include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
# include <arm_neon.h>
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
# include <intrin.h>
# if defined(PXL_HAS_MSVC_AVX2)
extern "C" uint64_t pxl_msvc_equal_avx2(const uint8_t*, const uint8_t*, uint32_t);
# endif
# if defined(PXL_HAS_MSVC_AVX512)
extern "C" uint64_t pxl_msvc_equal_avx512(const uint8_t*, const uint8_t*, uint32_t);
# endif
#endif
#if defined(_WIN32)
# define PXL_EXPORT __declspec(dllexport)
#else
# define PXL_EXPORT __attribute__((visibility("default")))
#endif

namespace {
constexpr uint32_t SCALE_BITS=12, TOTAL=1u<<SCALE_BITS, RANS_L=1u<<23;

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2")))
size_t equal_avx2(const uint8_t* a,const uint8_t* b,size_t n) {
    size_t i=0,count=0;
    for(;i+32<=n;i+=32) {
        const auto x=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+i));
        const auto y=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b+i));
        count+=size_t(__builtin_popcount(uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(x,y)))));
    }
    for(;i<n;++i) count+=(a[i]==b[i]);
    return count;
}
__attribute__((target("avx512f,avx512bw")))
size_t equal_avx512(const uint8_t* a,const uint8_t* b,size_t n) {
    size_t i=0,count=0;
    for(;i+64<=n;i+=64) {
        const auto x=_mm512_loadu_si512(reinterpret_cast<const void*>(a+i));
        const auto y=_mm512_loadu_si512(reinterpret_cast<const void*>(b+i));
        count+=size_t(__builtin_popcountll(uint64_t(_mm512_cmpeq_epi8_mask(x,y))));
    }
    return count+equal_avx2(a+i,b+i,n-i);
}
#endif
int simd_level() {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) return 3;
    if (__builtin_cpu_supports("avx2")) return 2;
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int regs[4] = {};
    __cpuid(regs, 1);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) return 0;
    const unsigned __int64 xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) return 0;
    __cpuidex(regs, 7, 0);
    const unsigned ebx = static_cast<unsigned>(regs[1]);
    const bool avx2 = (ebx & (1u << 5)) != 0;
    const bool avx512_os = (xcr0 & 0xE6) == 0xE6;
    const bool avx512bw = (ebx & (1u << 16)) && (ebx & (1u << 30));
# if defined(PXL_HAS_MSVC_AVX512)
    if (avx512_os && avx512bw) return 3;
# endif
# if defined(PXL_HAS_MSVC_AVX2)
    if (avx2) return 2;
# endif
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return 1;
#endif
    return 0;
}
size_t equal_count(const uint8_t* a,const uint8_t* b,size_t n) {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    const int level=simd_level();
    if(level>=3) return equal_avx512(a,b,n);
    if(level>=2) return equal_avx2(a,b,n);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    const int level=simd_level();
# if defined(PXL_HAS_MSVC_AVX512)
    if(level>=3) return size_t(pxl_msvc_equal_avx512(a,b,uint32_t(n)));
# endif
# if defined(PXL_HAS_MSVC_AVX2)
    if(level>=2) return size_t(pxl_msvc_equal_avx2(a,b,uint32_t(n)));
# endif
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    size_t i=0,count=0;
    for(;i+16<=n;i+=16) {
        uint8x16_t x=vld1q_u8(a+i),y=vld1q_u8(b+i),eq=vceqq_u8(x,y);
        uint8_t lanes[16]; vst1q_u8(lanes,eq);
        for(int k=0;k<16;++k) count+=(lanes[k]==255);
    }
    for(;i<n;++i) count+=(a[i]==b[i]);
    return count;
#else
    size_t count=0; for(size_t i=0;i<n;++i) count+=(a[i]==b[i]); return count;
#endif
}

uint32_t ctx_for(const uint8_t* out,uint32_t i,uint32_t row,uint32_t chn) {
    uint32_t col=i%row,ch=0,base=16;
    if(col<chn) ch=col;
    else {
        uint32_t q=col-chn,x=q/chn; ch=q%chn;
        uint8_t l=x?out[i-chn]:0, u=i>=row?out[i-row]:0;
        uint32_t cl=l==0?0:(l==1?1:(l==255?2:3));
        uint32_t cu=u==0?0:(u==1?1:(u==255?2:3));
        base=cl*4+cu;
    }
    return ch*17+base;
}

bool normalize(const std::array<uint64_t,256>& counts,std::array<uint16_t,256>& f) {
    uint64_t total=0; std::vector<uint32_t> symbols; std::array<uint64_t,256> rem{}; f.fill(0);
    for(uint32_t s=0;s<256;++s) if(counts[s]) {total+=counts[s];symbols.push_back(s);}
    if(!total) return false;
    uint32_t sum=0;
    for(uint32_t s:symbols) { uint64_t v=counts[s]*TOTAL; rem[s]=v%total; f[s]=uint16_t(std::max<uint64_t>(1,v/total)); sum+=f[s]; }
    if(sum<TOTAL) {
        std::stable_sort(symbols.begin(),symbols.end(),[&](uint32_t a,uint32_t b){return rem[a]!=rem[b]?rem[a]>rem[b]:a>b;});
        for(uint32_t i=0;i<TOTAL-sum;++i) ++f[symbols[i%symbols.size()]];
    } else if(sum>TOTAL) {
        uint32_t need=sum-TOTAL;
        while(need) {
            std::vector<uint32_t> can; for(uint32_t s:symbols) if(f[s]>1) can.push_back(s);
            if(can.empty()) return false;
            std::stable_sort(can.begin(),can.end(),[&](uint32_t a,uint32_t b){return rem[a]!=rem[b]?rem[a]<rem[b]:f[a]>f[b];});
            for(uint32_t s:can) {if(!need) break;--f[s];--need;}
        }
    }
    return true;
}
void varint(std::vector<uint8_t>& o,uint64_t v) { while(v>=128){o.push_back(uint8_t((v&127)|128));v>>=7;}o.push_back(uint8_t(v)); }
bool read_varint(const uint8_t* d,uint32_t n,uint32_t& p,uint64_t& v) {
    v=0;uint32_t shift=0;for(int i=0;i<10;++i){if(p>=n||shift>=64)return false;uint8_t b=d[p++];v|=uint64_t(b&127)<<shift;if(!(b&128))return true;shift+=7;}return false;
}
std::vector<uint8_t> table_bytes(const std::array<uint16_t,256>& f) {
    std::vector<uint8_t> dense{0}; dense.reserve(513);
    for(uint16_t x:f){dense.push_back(uint8_t(x));dense.push_back(uint8_t(x>>8));}
    std::vector<uint8_t> sparse{1}; std::vector<uint32_t> ids;
    for(uint32_t i=0;i<256;++i) if(f[i]) ids.push_back(i);
    sparse.push_back(uint8_t(ids.size()));
    for(uint32_t i:ids){sparse.push_back(uint8_t(i));varint(sparse,f[i]);}
    return sparse.size()<dense.size()?sparse:dense;
}

int mixed_encode(const uint8_t* data,uint32_t n,uint32_t w,uint32_t h,uint32_t chn,uint8_t* out,uint32_t cap,uint32_t* out_n) {
    if(!data||!out||!out_n||!w||!h||chn<1||chn>4) return -1;
    const uint32_t row=w*chn+chn, nctx=chn*17;
    if(uint64_t(row)*h!=n) return -2;
    std::vector<uint8_t> ids(n);
    std::vector<std::array<uint64_t,256>> local(nctx),channel(chn);
    std::array<uint64_t,256> global{};
    for(uint32_t i=0;i<n;++i){uint32_t c=ctx_for(data,i,row,chn);ids[i]=uint8_t(c);++local[c][data[i]];++global[data[i]];uint32_t col=i%row,ch=col<chn?col:(col-chn)%chn;++channel[ch][data[i]];}
    std::vector<std::array<uint16_t,256>> freq(nctx),cumul(nctx);
    std::vector<uint8_t> result{uint8_t(nctx)};
    for(uint32_t c=0;c<nctx;++c){std::array<uint64_t,256> combined{};uint32_t cc=c/17;for(uint32_t s=0;s<256;++s)combined[s]=4*local[c][s]+2*channel[cc][s]+global[s];
        if(!normalize(combined,freq[c])){result.push_back(0);result.push_back(0);continue;}
        uint32_t cm=0;for(uint32_t s=0;s<256;++s){cumul[c][s]=uint16_t(cm);cm+=freq[c][s];}
        auto t=table_bytes(freq[c]);if(t.size()>65535)return -3;result.push_back(uint8_t(t.size()));result.push_back(uint8_t(t.size()>>8));result.insert(result.end(),t.begin(),t.end());}
    uint64_t state=RANS_L;std::vector<uint8_t> reverse;reverse.reserve(n/4+32);
    for(uint32_t ri=n;ri>0;--ri){uint32_t i=ri-1,c=ids[i],s=data[i],f=freq[c][s];if(!f)return -4;uint64_t xmax=((uint64_t(RANS_L)>>SCALE_BITS)<<8)*f;while(state>=xmax){reverse.push_back(uint8_t(state));state>>=8;}state=((state/f)<<SCALE_BITS)+(state%f)+cumul[c][s];if(state>0xffffffffULL)return -5;}
    size_t total=result.size()+4+reverse.size();if(total>cap)return -6;for(int i=0;i<4;++i)result.push_back(uint8_t(state>>(i*8)));for(auto i=reverse.rbegin();i!=reverse.rend();++i)result.push_back(*i);
    std::memcpy(out,result.data(),result.size());*out_n=uint32_t(result.size());return 0;
}

int mixed_decode(const uint8_t* blob,uint32_t bn,uint32_t outn,uint32_t w,uint32_t h,uint32_t chn,uint8_t* out) {
    if(!blob||!out||!w||!h||chn<1||chn>4)return -1;uint32_t row=w*chn+chn,nctx=chn*17;if(uint64_t(row)*h!=outn||bn<1+nctx*2+4||blob[0]!=nctx)return -2;
    std::vector<std::array<uint16_t,256>> freq(nctx),cum(nctx);std::vector<std::array<uint8_t,TOTAL>> dec(nctx);uint32_t p=1;
    for(uint32_t c=0;c<nctx;++c){if(p+2>bn)return -3;uint32_t sz=blob[p]|(uint32_t(blob[p+1])<<8);p+=2;if(!sz)continue;if(p+sz>bn)return -4;uint32_t end=p+sz;uint8_t mode=blob[p++];
        if(mode==0){if(p+512!=end)return -5;for(uint32_t s=0;s<256;++s){freq[c][s]=uint16_t(blob[p]|(uint16_t(blob[p+1])<<8));p+=2;}}
        else if(mode==1){if(p>=end)return -6;uint32_t count=blob[p++];if(!count)count=256;for(uint32_t k=0;k<count;++k){if(p>=end)return -7;uint32_t s=blob[p++];uint64_t v=0;uint32_t sh=0;bool ok=false;for(int j=0;j<5;++j){if(p>=end||sh>=32)return -8;uint8_t b=blob[p++];v|=uint64_t(b&127)<<sh;if(!(b&128)){ok=true;break;}sh+=7;}if(!ok||s>255||!v||v>TOTAL||freq[c][s])return -9;freq[c][s]=uint16_t(v);}}
        else return -10;
        if(p!=end)return -11;

        // Validate the complete frequency table before populating the fixed-size
        // 4096-entry decoder table. A malformed dense table can otherwise make
        // sum+j exceed TOTAL and write past dec[c] before the old final check.
        uint32_t sum=0;
        for(uint32_t s=0;s<256;++s){
            const uint32_t frequency=freq[c][s];
            if(frequency>TOTAL-sum)return -12;
            sum+=frequency;
        }
        if(sum!=TOTAL)return -12;

        // The table is now proven to occupy exactly [0, TOTAL).
        sum=0;
        for(uint32_t s=0;s<256;++s){
            cum[c][s]=uint16_t(sum);
            for(uint32_t j=0;j<freq[c][s];++j)dec[c][sum+j]=uint8_t(s);
            sum+=freq[c][s];
        }
    }
    if(p+4>bn)return -13;uint64_t state=uint64_t(blob[p])|(uint64_t(blob[p+1])<<8)|(uint64_t(blob[p+2])<<16)|(uint64_t(blob[p+3])<<24);p+=4;if(state<RANS_L)return -14;
    for(uint32_t i=0;i<outn;++i){uint32_t c=ctx_for(out,i,row,chn),slot=uint32_t(state&(TOTAL-1)),s=dec[c][slot],f=freq[c][s];if(!f)return -15;state=uint64_t(f)*(state>>SCALE_BITS)+slot-cum[c][s];while(state<RANS_L){if(p>=bn)return -16;state=(state<<8)|blob[p++];}out[i]=uint8_t(s);}return p==bn?0:-17;
}

void write_v(std::vector<uint8_t>& o,uint64_t v){while(v>=128){o.push_back(uint8_t((v&127)|128));v>>=7;}o.push_back(uint8_t(v));}
std::vector<uint8_t> delta_stream(const std::vector<uint8_t>& a,const std::vector<uint8_t>& b){std::vector<uint8_t> o;size_t p=0;while(p<a.size()){size_t z=0;while(p<a.size()&&uint8_t(a[p]-b[p])==0){++p;++z;}size_t s=p;while(p<a.size()&&uint8_t(a[p]-b[p])!=0)++p;write_v(o,z);write_v(o,p-s);for(size_t i=s;i<p;++i)o.push_back(uint8_t(a[i]-b[i]));}return o;}
bool delta_restore(const uint8_t* d,uint32_t n,uint32_t& p,const std::vector<uint8_t>& ref,std::vector<uint8_t>& out){out.resize(ref.size());size_t made=0;while(made<ref.size()){uint64_t z=0,l=0;if(!read_varint(d,n,p,z)||!read_varint(d,n,p,l)||(!z&&!l)||z>ref.size()-made)return false;for(uint64_t i=0;i<z;++i){out[made]=ref[made];++made;}if(l>ref.size()-made||l>n-p)return false;for(uint64_t i=0;i<l;++i){out[made]=uint8_t(ref[made]+d[p++]);++made;}}return true;}
struct Item{uint32_t w=0,h=0;std::vector<uint8_t> bytes;};
class VarBlocks{
public:
    VarBlocks(uint32_t ch,uint32_t lim):ch_(ch),lim_(lim){if(ch<1||ch>4||lim<64)throw std::invalid_argument("bad variable block state");}
    int encode(const uint8_t* raw,uint32_t w,uint32_t h,uint8_t* out,uint32_t cap,uint32_t* outn){
        if(!raw||!out||!outn||!w||!h)return -1;std::vector<uint8_t> tok,cover(size_t((w+7)/8)*((h+7)/8),0);uint32_t gx=(w+7)/8;
        for(uint32_t y=0;y<h;y+=8)for(uint32_t x=0;x<w;x+=8){if(cover[size_t(y/8)*gx+x/8])continue;uint32_t bestop=0,bestsize=0;uint64_t bestid=0;std::vector<uint8_t> bestdelta;size_t bestcost=std::numeric_limits<size_t>::max();
            for(uint32_t size:{32u,16u,8u}){if(x+size>w||y+size>h)continue;bool overlap=false;for(uint32_t yy=y;yy<y+size;yy+=8)for(uint32_t xx=x;xx<x+size;xx+=8)if(cover[size_t(yy/8)*gx+xx/8])overlap=true;if(overlap)continue;auto b=extract(raw,w,x,y,size,size);auto exact=find(size,size,b);if(exact){size_t c=2+varint_size(next_-*exact);if(c<bestcost){bestcost=c;bestop=1;bestsize=size;bestid=*exact;bestdelta.clear();}continue;}if(size==8)continue;
                auto it=shapes_.find(key(size,size));if(it==shapes_.end())continue;for(auto q=it->second.rbegin();q!=it->second.rend();++q){const Item* r=get(*q);if(!r)continue;size_t eq=equal_count(b.data(),r->bytes.data(),b.size());if(eq*5<b.size())continue;auto d=delta_stream(b,r->bytes);size_t cost=2+varint_size(next_-*q)+d.size();if(cost<bestcost&&cost<b.size()){bestcost=cost;bestop=2;bestsize=size;bestid=*q;bestdelta=std::move(d);}}}
            uint32_t bw=8,bh=8;std::vector<uint8_t> b;if(!bestop){bw=std::min(8u,w-x);bh=std::min(8u,h-y);b=extract(raw,w,x,y,bw,bh);tok.push_back(0);tok.push_back(0);tok.insert(tok.end(),b.begin(),b.end());}else{bw=bh=bestsize;b=extract(raw,w,x,y,bw,bh);tok.push_back(uint8_t(bestop));tok.push_back(uint8_t(bestsize==32?2:(bestsize==16?1:0)));write_v(tok,next_-bestid);if(bestop==2)tok.insert(tok.end(),bestdelta.begin(),bestdelta.end());}
            for(uint32_t yy=y;yy<y+bh;yy+=8)for(uint32_t xx=x;xx<x+bw;xx+=8)cover[size_t(yy/8)*gx+xx/8]=1;add_region(bw,bh,b);if(tok.size()>cap)return -2;}
        std::memcpy(out,tok.data(),tok.size());*outn=uint32_t(tok.size());return 0;}
    int update(const uint8_t* raw,uint32_t w,uint32_t h){if(!raw||!w||!h)return -1;for(uint32_t s:{32u,16u,8u})for(uint32_t y=0;y<h;y+=s)for(uint32_t x=0;x<w;x+=s){uint32_t bw=std::min(s,w-x),bh=std::min(s,h-y);if(s>8&&(bw!=s||bh!=s))continue;add(bw,bh,extract(raw,w,x,y,bw,bh));}return 0;}
    int has_large(const uint8_t* raw,uint32_t w,uint32_t h){if(!raw)return 0;for(uint32_t s:{32u,16u})if(w>=s&&h>=s)for(uint32_t y=0;y+s<=h;y+=s)for(uint32_t x=0;x+s<=w;x+=s){auto b=extract(raw,w,x,y,s,s);if(find(s,s,b))return 1;auto i=shapes_.find(key(s,s));if(i!=shapes_.end())for(auto q=i->second.rbegin();q!=i->second.rend();++q){const Item* r=get(*q);if(!r)continue;if(equal_count(b.data(),r->bytes.data(),b.size())*5>=b.size()&&delta_stream(b,r->bytes).size()+3<b.size())return 1;}}return 0;}
    int decode(const uint8_t* t,uint32_t tn,uint32_t w,uint32_t h,uint8_t* out,uint32_t outn){if(!t||!out||!w||!h||uint64_t(w)*h*ch_!=outn)return -1;uint32_t p=0,gx=(w+7)/8;std::vector<uint8_t> cover(size_t(gx)*((h+7)/8),0);for(uint32_t y=0;y<h;y+=8)for(uint32_t x=0;x<w;x+=8){if(cover[size_t(y/8)*gx+x/8])continue;if(p+2>tn)return -2;uint8_t op=t[p++],code=t[p++];if(code>2)return -3;uint32_t s=code==0?8:(code==1?16:32),bw=std::min(8u,w-x),bh=std::min(8u,h-y);std::vector<uint8_t>b;if(op==0){if(code)return -4;size_t n=size_t(bw)*bh*ch_;if(n>tn-p)return -5;b.assign(t+p,t+p+n);p+=uint32_t(n);}else if(op==1||op==2){if(x+s>w||y+s>h)return -6;uint64_t dist=0;if(!read_varint(t,tn,p,dist)||!dist||dist>next_)return -7;const Item*r=get(next_-dist);if(!r||r->w!=s||r->h!=s)return -8;bw=bh=s;if(op==1)b=r->bytes;else if(!delta_restore(t,tn,p,r->bytes,b))return -9;}else return -10;size_t q=0;for(uint32_t yy=0;yy<bh;++yy){size_t n=size_t(bw)*ch_;std::memcpy(out+(uint64_t(y+yy)*w+x)*ch_,b.data()+q,n);q+=n;}for(uint32_t yy=y;yy<y+bh;yy+=8)for(uint32_t xx=x;xx<x+bw;xx+=8)cover[size_t(yy/8)*gx+xx/8]=1;add_region(bw,bh,b);}return p==tn?0:-11;}
private:
    uint32_t ch_,lim_;uint64_t next_=0,base_=0;std::deque<Item> items_;std::unordered_map<uint64_t,std::vector<uint64_t>> heads_;std::unordered_map<uint32_t,std::deque<uint64_t>> shapes_;
    uint32_t key(uint32_t w,uint32_t h)const{return(w<<16)|h;}uint64_t hash(uint32_t w,uint32_t h,const std::vector<uint8_t>&b)const{uint64_t v=1469598103934665603ULL;v^=w;v*=1099511628211ULL;v^=h;v*=1099511628211ULL;for(uint8_t x:b){v^=x;v*=1099511628211ULL;}return v?v:1;}
    uint32_t varint_size(uint64_t v)const{uint32_t n=1;while(v>=128){++n;v>>=7;}return n;}
    const Item* get(uint64_t id)const{if(id<base_||id>=next_)return nullptr;uint64_t q=id-base_;return q<items_.size()?&items_[size_t(q)]:nullptr;}
    std::vector<uint8_t> extract(const uint8_t*r,uint32_t w,uint32_t x,uint32_t y,uint32_t bw,uint32_t bh)const{std::vector<uint8_t>b(size_t(bw)*bh*ch_);size_t p=0;for(uint32_t yy=0;yy<bh;++yy){size_t n=size_t(bw)*ch_;std::memcpy(b.data()+p,r+(uint64_t(y+yy)*w+x)*ch_,n);p+=n;}return b;}
    std::optional<uint64_t> find(uint32_t w,uint32_t h,const std::vector<uint8_t>&b)const{auto i=heads_.find(hash(w,h,b));if(i==heads_.end())return{};for(auto q=i->second.rbegin();q!=i->second.rend();++q){auto r=get(*q);if(r&&r->w==w&&r->h==h&&r->bytes==b)return *q;}return{};}
    void add(uint32_t w,uint32_t h,std::vector<uint8_t>&&b){uint64_t id=next_++;auto k=hash(w,h,b);items_.push_back(Item{w,h,std::move(b)});auto&q=heads_[k];q.push_back(id);if(q.size()>16)q.erase(q.begin());auto&sh=shapes_[key(w,h)];sh.push_back(id);if(sh.size()>64)sh.pop_front();if(items_.size()>lim_){items_.pop_front();++base_;}}
    void add_region(uint32_t w,uint32_t h,const std::vector<uint8_t>&b){add(w,h,std::vector<uint8_t>(b));for(uint32_t s:{16u,8u})if(w>=s&&h>=s)for(uint32_t y=0;y+s<=h;y+=s)for(uint32_t x=0;x+s<=w;x+=s){std::vector<uint8_t>sub(size_t(s)*s*ch_);size_t p=0;for(uint32_t yy=0;yy<s;++yy){size_t n=size_t(s)*ch_;std::memcpy(sub.data()+p,b.data()+(uint64_t(y+yy)*w+x)*ch_,n);p+=n;}add(s,s,std::move(sub));}}
};
} // namespace

extern "C" {
PXL_EXPORT int pxl_v8_simd_level(){return simd_level();}
PXL_EXPORT uint64_t pxl_v8_equal_count(const uint8_t*a,const uint8_t*b,uint32_t n){return a&&b?uint64_t(equal_count(a,b,n)):0;}
PXL_EXPORT int pxl_v8_mixed_encode(const uint8_t*d,uint32_t n,uint32_t w,uint32_t h,uint32_t c,uint8_t*out,uint32_t cap,uint32_t*outn){try{return mixed_encode(d,n,w,h,c,out,cap,outn);}catch(...){return -100;}}
PXL_EXPORT int pxl_v8_mixed_decode(const uint8_t*d,uint32_t n,uint32_t outn,uint32_t w,uint32_t h,uint32_t c,uint8_t*out){try{return mixed_decode(d,n,outn,w,h,c,out);}catch(...){return -100;}}
PXL_EXPORT void* pxl_v8_var_create(uint32_t c,uint32_t l){try{return new VarBlocks(c,l);}catch(...){return nullptr;}}
PXL_EXPORT void* pxl_v8_var_clone(void*s){try{return s?new VarBlocks(*static_cast<VarBlocks*>(s)):nullptr;}catch(...){return nullptr;}}
PXL_EXPORT void pxl_v8_var_destroy(void*s){delete static_cast<VarBlocks*>(s);}
PXL_EXPORT int pxl_v8_var_encode(void*s,const uint8_t*d,uint32_t w,uint32_t h,uint8_t*out,uint32_t cap,uint32_t*outn){try{return s?static_cast<VarBlocks*>(s)->encode(d,w,h,out,cap,outn):-1;}catch(...){return -100;}}
PXL_EXPORT int pxl_v8_var_update(void*s,const uint8_t*d,uint32_t w,uint32_t h){try{return s?static_cast<VarBlocks*>(s)->update(d,w,h):-1;}catch(...){return -100;}}
PXL_EXPORT int pxl_v8_var_has_large(void*s,const uint8_t*d,uint32_t w,uint32_t h){try{return s?static_cast<VarBlocks*>(s)->has_large(d,w,h):0;}catch(...){return 0;}}
PXL_EXPORT int pxl_v8_var_decode(void*s,const uint8_t*t,uint32_t tn,uint32_t w,uint32_t h,uint8_t*out,uint32_t outn){try{return s?static_cast<VarBlocks*>(s)->decode(t,tn,w,h,out,outn):-1;}catch(...){return -100;}}
}
