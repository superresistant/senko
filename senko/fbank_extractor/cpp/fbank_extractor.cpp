#include "fbank_extractor.h"
#include "feature_computer.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

struct WavInfo {
    int sample_rate = 0;
    int channels = 0;
    int bits_per_sample = 0;
    int audio_format = 0;
    int64_t data_offset = 0;
    int64_t data_size = 0;
};

static uint16_t read_u16_le(const uint8_t* data, size_t offset) {
    return static_cast<uint16_t>(data[offset]) |
           static_cast<uint16_t>(data[offset + 1]) << 8;
}

static uint32_t read_u32_le(const uint8_t* data, size_t offset) {
    return static_cast<uint32_t>(data[offset]) |
           static_cast<uint32_t>(data[offset + 1]) << 8 |
           static_cast<uint32_t>(data[offset + 2]) << 16 |
           static_cast<uint32_t>(data[offset + 3]) << 24;
}

static bool read_fully(int fd, void* buffer, size_t count, off_t offset) {
    uint8_t* dst = static_cast<uint8_t*>(buffer);
    size_t total = 0;
    while (total < count) {
        ssize_t n = pread(fd, dst + total, count - total, offset + static_cast<off_t>(total));
        if (n <= 0) {
            return false;
        }
        total += static_cast<size_t>(n);
    }
    return true;
}

static bool parse_wav_header(int fd, WavInfo& info) {
    struct stat st;
    if (fstat(fd, &st) != 0) {
        return false;
    }
    const int64_t file_size = static_cast<int64_t>(st.st_size);
    if (file_size < 12) {
        return false;
    }

    uint8_t header[12];
    if (!read_fully(fd, header, sizeof(header), 0)) {
        return false;
    }

    if (std::string(reinterpret_cast<char*>(header), 4) != "RIFF" ||
        std::string(reinterpret_cast<char*>(header + 8), 4) != "WAVE") {
        return false;
    }

    bool fmt_found = false;
    bool data_found = false;

    int64_t offset = 12;
    while (offset + 8 <= file_size) {
        uint8_t chunk_header[8];
        if (!read_fully(fd, chunk_header, sizeof(chunk_header), offset)) {
            return false;
        }

        const std::string chunk_id(reinterpret_cast<char*>(chunk_header), 4);
        const uint32_t chunk_size = read_u32_le(chunk_header, 4);
        const int64_t chunk_data_offset = offset + 8;

        if (chunk_id == "fmt ") {
            const uint32_t fmt_size = std::min<uint32_t>(chunk_size, 16);
            if (fmt_size < 16) {
                return false;
            }
            uint8_t fmt[16];
            if (!read_fully(fd, fmt, fmt_size, chunk_data_offset)) {
                return false;
            }
            info.audio_format = static_cast<int>(read_u16_le(fmt, 0));
            info.channels = static_cast<int>(read_u16_le(fmt, 2));
            info.sample_rate = static_cast<int>(read_u32_le(fmt, 4));
            info.bits_per_sample = static_cast<int>(read_u16_le(fmt, 14));
            fmt_found = true;
        } else if (chunk_id == "data") {
            info.data_offset = chunk_data_offset;
            if (chunk_size == 0) {
                info.data_size = file_size - info.data_offset;
            } else {
                info.data_size = static_cast<int64_t>(chunk_size);
            }
            data_found = true;
        }

        const int64_t padded_size = chunk_size + (chunk_size % 2);
        offset += 8 + padded_size;

        if (fmt_found && data_found) {
            break;
        }
    }

    if (!fmt_found || !data_found) {
        return false;
    }

    if (info.data_offset + info.data_size > file_size) {
        return false;
    }

    return true;
}

class WavStreamReader {
public:
    explicit WavStreamReader(const std::string& path) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            return;
        }
        if (!parse_wav_header(fd_, info_)) {
            close(fd_);
            fd_ = -1;
            return;
        }
        bytes_per_sample_ = info_.bits_per_sample / 8;
        if (bytes_per_sample_ <= 0 || info_.channels <= 0) {
            close(fd_);
            fd_ = -1;
            return;
        }
        bytes_per_frame_ = bytes_per_sample_ * info_.channels;
        total_frames_ = static_cast<size_t>(info_.data_size / bytes_per_frame_);
    }

    ~WavStreamReader() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    bool valid() const { return fd_ >= 0; }
    size_t num_samples() const { return total_frames_; }
    int channels() const { return info_.channels; }
    int bits_per_sample() const { return info_.bits_per_sample; }
    int audio_format() const { return info_.audio_format; }

    bool read_samples(size_t start_frame, size_t frame_count, std::vector<float>& out) const {
        if (!valid()) {
            return false;
        }
        if (frame_count == 0) {
            out.clear();
            return true;
        }
        if (start_frame >= total_frames_) {
            out.clear();
            return true;
        }

        const size_t available = total_frames_ - start_frame;
        const size_t to_read = std::min(frame_count, available);
        out.resize(to_read);

        const off_t byte_offset = static_cast<off_t>(info_.data_offset) +
            static_cast<off_t>(start_frame * bytes_per_frame_);
        const size_t byte_count = to_read * bytes_per_frame_;

        const float scale = 1.0f / 32768.0f;

        switch (info_.bits_per_sample) {
            case 8: {
                std::vector<int8_t> interleaved(to_read * info_.channels);
                if (!read_fully(fd_, interleaved.data(), byte_count, byte_offset)) {
                    return false;
                }
                for (size_t i = 0; i < to_read; ++i) {
                    out[i] = static_cast<float>(interleaved[i * info_.channels]) * scale;
                }
                break;
            }
            case 16: {
                std::vector<int16_t> interleaved(to_read * info_.channels);
                if (!read_fully(fd_, interleaved.data(), byte_count, byte_offset)) {
                    return false;
                }
                for (size_t i = 0; i < to_read; ++i) {
                    out[i] = static_cast<float>(interleaved[i * info_.channels]) * scale;
                }
                break;
            }
            case 32: {
                if (info_.audio_format == 1) { // PCM int32
                    std::vector<int32_t> interleaved(to_read * info_.channels);
                    if (!read_fully(fd_, interleaved.data(), byte_count, byte_offset)) {
                        return false;
                    }
                    for (size_t i = 0; i < to_read; ++i) {
                        out[i] = static_cast<float>(interleaved[i * info_.channels]) * scale;
                    }
                } else if (info_.audio_format == 3) { // IEEE float
                    std::vector<float> interleaved(to_read * info_.channels);
                    if (!read_fully(fd_, interleaved.data(), byte_count, byte_offset)) {
                        return false;
                    }
                    for (size_t i = 0; i < to_read; ++i) {
                        out[i] = interleaved[i * info_.channels];
                    }
                } else {
                    return false;
                }
                break;
            }
            default:
                return false;
        }

        return true;
    }

private:
    int fd_ = -1;
    WavInfo info_;
    size_t bytes_per_sample_ = 0;
    size_t bytes_per_frame_ = 0;
    size_t total_frames_ = 0;
};

}  // namespace

FbankExtractor::FbankExtractor() {}

FbankResult FbankExtractor::extract_features(const std::string& wav_path, const std::vector<std::pair<float, float>>& subsegments) {

    /*═════════════╗
    ║  Load audio  ║
    ╚═════════════*/

    WavStreamReader wav_reader(wav_path);
    if (!wav_reader.valid()) {
        return {{}, {}, {}};
    }
    const size_t total_samples = wav_reader.num_samples();

    /*════════════════════════════════════════════╗
    ║  Fbank feature extraction (multi-threaded)  ║
    ╚════════════════════════════════════════════*/

    FeatureComputer fc;

    // Each feature is 80 mel bins ("height") by up to ~150 frames ("width")
    constexpr size_t mel_bins = 80;
    constexpr size_t max_frames_per_subseg = 150;
    constexpr size_t sample_rate = 16000;

    // Allocate a single large buffer for all subsegments
    std::vector<float> big_features(subsegments.size() * max_frames_per_subseg * mel_bins, 0.f);

    // For each subsegment, track (offset_in_big_features, frames_produced)
    std::vector<std::pair<size_t, size_t>> feature_indices(subsegments.size());

    // Track frame counts for each subsegment (this is what we'll return)
    std::vector<size_t> frames_per_subsegment(subsegments.size());
    std::vector<size_t> subsegment_offsets(subsegments.size());

    // Multi-threading setup
    const unsigned int feat_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> feat_workers;
    feat_workers.reserve(feat_threads);

    // Use an atomic offset so each thread knows where to write in big_features
    std::atomic_size_t global_offset{0};

    // Thread-local buffers for streaming reads and padding short segments
    thread_local static std::vector<float> segment_buffer;
    thread_local static std::vector<float> padded_buffer;

    auto feat_worker = [&](size_t first, size_t last) {
        for (size_t i = first; i < last; ++i) {
            const auto& sub = subsegments[i];
            size_t sample_start = static_cast<size_t>(sub.first * sample_rate);
            size_t sample_len = static_cast<size_t>((sub.second - sub.first) * sample_rate);

            if (sample_len == 0) sample_len = 1;
            if (sample_start >= total_samples) {
                sample_len = 0;
            } else if (sample_start + sample_len > total_samples) {
                sample_len = total_samples - sample_start;
            }

            const size_t min_len = 400;  // Ensure a minimum of 400 samples
            std::span<float> wav_span;

            if (sample_len < min_len) {
                if (padded_buffer.size() < min_len) padded_buffer.resize(min_len, 0.f);
                std::fill(padded_buffer.begin(), padded_buffer.begin() + min_len, 0.f);
                if (sample_len > 0 && wav_reader.read_samples(sample_start, sample_len, segment_buffer)) {
                    std::copy(segment_buffer.begin(),
                             segment_buffer.begin() + sample_len,
                             padded_buffer.begin());
                }
                // Remaining samples in padded_buffer stay zero
                wav_span = std::span<float>(padded_buffer.data(), min_len);
            } else {
                if (wav_reader.read_samples(sample_start, sample_len, segment_buffer)) {
                    wav_span = std::span<float>(segment_buffer.data(), sample_len);
                } else {
                    if (padded_buffer.size() < min_len) padded_buffer.resize(min_len, 0.f);
                    std::fill(padded_buffer.begin(), padded_buffer.begin() + min_len, 0.f);
                    wav_span = std::span<float>(padded_buffer.data(), min_len);
                }
            }

            // Compute FBank features => 2D: (frames x mel_bins)
            auto feat2d = fc.compute_feature(wav_span);
            const size_t frames = feat2d.size(); // #frames generated

            // Store frame count for this subsegment
            frames_per_subsegment[i] = frames;

            // Claim a chunk in big_features for these features
            const size_t my_off = global_offset.fetch_add(frames * mel_bins);
            feature_indices[i] = {my_off, frames};
            subsegment_offsets[i] = my_off;

            // Flatten-copy (frame by frame) into big_features
            size_t write_ptr = my_off;
            for (const auto& frame : feat2d) {
                // Each frame => mel_bins floats
                std::copy(frame.begin(), frame.end(),
                         big_features.begin() + static_cast<long>(write_ptr));
                write_ptr += mel_bins;
            }
        }
    };

    // Distribute subsegments across threads
    const size_t sub_per_thread = (subsegments.size() + feat_threads - 1) / feat_threads;
    size_t idx0 = 0;
    for (unsigned int t = 0; t < feat_threads; ++t) {
        const size_t idx1 = std::min(idx0 + sub_per_thread, subsegments.size());
        feat_workers.emplace_back(feat_worker, idx0, idx1);
        idx0 = idx1;
    }

    for (auto& th : feat_workers) th.join();

    // Shrink big_features to actual usage
    const size_t used_size = global_offset.load();
    big_features.resize(used_size);
    big_features.shrink_to_fit();

    return {big_features, frames_per_subsegment, subsegment_offsets};
}
