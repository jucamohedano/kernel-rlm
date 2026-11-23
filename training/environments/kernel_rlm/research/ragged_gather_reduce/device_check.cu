// Trusted author-controls checker. Link exactly one author implementation.
// This is not a sandbox or a trustworthy verdict for an untrusted linked object.
#include "contract.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
void require_wire(bool ok, const char* field) {
    if (!ok) {
        std::cerr << "invalid wire: " << field << '\n';
        std::exit(2);
    }
}
int64_t number(const char* field, int64_t lo, int64_t hi) {
    int64_t x;
    require_wire(bool(std::cin >> x) && x >= lo && x <= hi, field);
    return x;
}
template <class T>
std::vector<T> read_array(const char* name, int n) {
    std::vector<T> a(n);
    for (T& x : a) x = static_cast<T>(number(name, std::numeric_limits<T>::min(),
                                              std::numeric_limits<T>::max()));
    return a;
}
void check_cuda(cudaError_t rc, const char* expression, int line) {
    if (rc != cudaSuccess) {
        std::cerr << "CUDA error at " << line << " (" << expression << "): "
                  << cudaGetErrorString(rc) << '\n';
        std::exit(3);
    }
}
#define CK(call) check_cuda((call), #call, __LINE__)

// Non-null, naturally aligned interior pointer even for a logical length of 0.
// Both guards, all inputs and every output slot are checked on readback.
template <class T>
class Guarded {
    static constexpr size_t PAD = 17;
    const char* name;
    T* storage = nullptr;
    std::vector<T> expected;
    std::vector<T> initial;
public:
    Guarded(const char* label, const std::vector<T>& payload, cudaStream_t stream,
            bool output = false) : name(label), expected(2 * PAD + payload.size(), T(0x523b)) {
        std::fill_n(expected.begin(), PAD, T(0x318c));
        std::copy(payload.begin(), payload.end(), expected.begin() + PAD);
        initial = expected;
        // An untouched output must fail even when its expected value is zero.
        if (output) {
            for (size_t i = PAD; i < PAD + payload.size(); ++i)
                initial[i] = expected[i] == T(0x6195) ? T(0x18d6) : T(0x6195);
        }
        const size_t bytes = expected.size() * sizeof(T);
        CK(cudaMalloc(reinterpret_cast<void**>(&storage), bytes));
        CK(cudaMemcpyAsync(storage, initial.data(), bytes, cudaMemcpyHostToDevice, stream));
    }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    ~Guarded() { CK(cudaFree(storage)); }
    T* data() const { return storage + PAD; }
    void verify(cudaStream_t stream) const {
        std::vector<T> got(expected.size());
        CK(cudaMemcpyAsync(got.data(), storage, got.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] != expected[i]) {
                std::cerr << "mismatch " << name << '[' << (int64_t(i) - int64_t(PAD))
                          << "]: expected " << expected[i] << ", got " << got[i] << '\n';
                std::exit(1);
            }
        }
    }
};
}  // namespace

int main(int argc, char** argv) {
    const bool timing = argc == 2 && std::string(argv[1]) == "--time";
    if (argc != 1 && !timing) {
        std::cerr << "usage: device_check [--time] < fixture\n";
        return 2;
    }
    constexpr int OWNER = 262144;
    const int rows = number("rows", 0, 4096);
    const int base = number("value_base", 0, OWNER);
    const int nv = number("nvalues", 0, OWNER - base);
    const int nw = number("nwords", 0, OWNER / 32);
    const auto offs = read_array<int32_t>("row_offsets", rows + 1);
    require_wire(offs.front() >= 0 && offs.back() <= OWNER &&
                 std::is_sorted(offs.begin(), offs.end()), "row_offsets range/order");
    const int n = offs.back() - offs.front();
    require_wire(n <= 65536 && nw == (offs.back() + 31) / 32, "entry/word length");
    const auto ids = read_array<int32_t>("gather_index", n);
    const auto mask = read_array<uint32_t>("keep_words", nw);
    const auto vals = read_array<int16_t>("values", nv);
    const auto sums = read_array<int32_t>("expected sums", rows);
    const auto counts = read_array<int32_t>("expected counts", rows);
    std::cin >> std::ws;
    require_wire(std::cin.eof(), "trailing data");
    for (int i = 0; i < n; ++i) {
        const int p = offs.front() + i;
        if ((mask[p / 32] >> (p % 32)) & 1u)
            require_wire(ids[i] >= base && ids[i] < base + nv, "kept value coordinate");
    }

    cudaStream_t stream;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Guarded<int32_t> offsets("row_offsets", offs, stream), gather("gather_index", ids, stream);
    Guarded<uint32_t> words("keep_words", mask, stream);
    Guarded<int16_t> values("values", vals, stream);
    Guarded<int32_t> out("row_sums", sums, stream, true), cnt("row_counts", counts, stream, true);
    auto invoke = [&]() {
        CK(ragged_gather_reduce(rows, offsets.data(), gather.data(), words.data(), base,
                                values.data(), out.data(), cnt.data(), stream));
    };
    auto verify = [&]() {
        out.verify(stream);
        cnt.verify(stream);
        offsets.verify(stream);
        gather.verify(stream);
        words.verify(stream);
        values.verify(stream);
    };
    invoke();
    verify();
    if (timing && rows != 0) {
        // Hot fixture; stream work only. Not a qualified performance threshold.
        constexpr int WARMUP = 10, REPEATS = 100, BATCHES = 5;
        cudaEvent_t start, stop;
        CK(cudaEventCreate(&start));
        CK(cudaEventCreate(&stop));
        for (int i = 0; i < WARMUP; ++i) invoke();
        std::array<float, BATCHES> us;
        for (float& t : us) {
            CK(cudaEventRecord(start, stream));
            for (int j = 0; j < REPEATS; ++j) invoke();
            CK(cudaEventRecord(stop, stream));
            CK(cudaEventSynchronize(stop));
            float ms;
            CK(cudaEventElapsedTime(&ms, start, stop));
            t = ms * 1000 / REPEATS;
        }
        CK(cudaEventDestroy(start));
        CK(cudaEventDestroy(stop));
        verify();
        std::cout << "unqualified_hot_batch_mean_us";
        for (float t : us) std::cout << ' ' << t;
        std::cout << '\n';
    }
    CK(cudaStreamDestroy(stream));
    std::cout << "match\n";
    return 0;
}
