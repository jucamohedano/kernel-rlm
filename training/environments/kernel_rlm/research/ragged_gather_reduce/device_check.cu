// Capture observations from one linked implementation. Same-process code is
// not isolated and can falsify observations, diagnostics, or exit status.
#include "contract.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
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
template <class T>
class Guarded {
    static constexpr size_t PAD = 17;
    const char* name;
    T* storage = nullptr;
    std::vector<T> initial;
public:
    Guarded(const char* label, const std::vector<T>& payload, cudaStream_t stream)
        : name(label), initial(2 * PAD + payload.size(), T(0x523b)) {
        std::fill_n(initial.begin(), PAD, T(0x318c));
        std::copy(payload.begin(), payload.end(), initial.begin() + PAD);
        const size_t bytes = initial.size() * sizeof(T);
        CK(cudaMalloc(reinterpret_cast<void**>(&storage), bytes));
        CK(cudaMemcpyAsync(storage, initial.data(), bytes, cudaMemcpyHostToDevice, stream));
    }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    ~Guarded() { CK(cudaFree(storage)); }
    T* data() const { return storage + PAD; }
    // Check inputs in full. For outputs, check both guards and capture the interior.
    void readback(cudaStream_t stream, std::vector<T>* output = nullptr) const {
        std::vector<T> got(initial.size());
        CK(cudaMemcpyAsync(got.data(), storage, got.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        for (size_t i = 0; i < got.size(); ++i) {
            if (output && i >= PAD && i < got.size() - PAD) continue;
            if (got[i] != initial[i]) {
                std::cerr << "mismatch " << name << '[' << (int64_t(i) - int64_t(PAD))
                          << "]: expected " << initial[i] << ", got " << got[i] << '\n';
                std::exit(1);
            }
        }
        if (output) output->assign(got.begin() + PAD, got.end() - PAD);
    }
};

struct Observation {
    std::vector<int32_t> sums, counts;
};
void print_line(const std::vector<int32_t>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) std::cout << ' ';
        std::cout << values[i];
    }
    std::cout << '\n';
}
}  // namespace

int main(int argc, char**) {
    if (argc != 1) {
        std::cerr << "usage: device_check < fixture\n";
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
    auto capture = [&](int32_t sentinel) {
        const std::vector<int32_t> fill(rows, sentinel);
        Guarded<int32_t> out("row_sums", fill, stream), cnt("row_counts", fill, stream);
        CK(ragged_gather_reduce(rows, offsets.data(), gather.data(), words.data(), base,
                                values.data(), out.data(), cnt.data(), stream));
        offsets.readback(stream);
        gather.readback(stream);
        words.readback(stream);
        values.readback(stream);
        Observation obs;
        out.readback(stream, &obs.sums);
        cnt.readback(stream, &obs.counts);
        return obs;
    };
    // A slot untouched on both runs cannot match the same reference twice.
    const std::array<Observation, 2> runs{capture(0x6195), capture(0x18d6)};
    CK(cudaStreamDestroy(stream));
    std::cout << "ragged-observation-v1\n";
    for (const auto& run : runs) {
        print_line(run.sums);
        print_line(run.counts);
    }
    if (!std::cout.flush()) {
        std::cerr << "failed to write observation\n";
        return 1;
    }
    return 0;
}
